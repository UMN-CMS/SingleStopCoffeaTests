import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import hist
import pickle
import uproot
from pathlib import Path
from coffea.analysis_tools import PackedSelection
from coffea import processor
import matplotlib.pyplot as plt
import numpy as np
import itertools
from coffea.processor import accumulate
from dask.distributed import Client


def is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak._ext.ArrayType):
        if isinstance(t.type, ak._ext.PrimitiveType):
            return True
        if isinstance(t.type, ak._ext.ListType) and isinstance(
            t.type.type, ak._ext.PrimitiveType
        ):
            return True
    return False


def uproot_writeable(events):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    for bname in events.fields:
        if events[bname].fields:
            out[bname] = ak.zip(
                {
                    n: ak.packed(ak.without_parameters(events[bname][n]))
                    for n in events[bname].fields
                    if is_rootcompat(events[bname][n])
                }
            )
        else:
            out[bname] = ak.packed(ak.without_parameters(events[bname]))
    return out


dataset_axis = hist.axis.StrCategory(
    [], growth=True, name="dataset", label="Primary dataset"
)

mass_axis = hist.axis.Regular(200, 0, 2000, name="mass", label=r"$m$ [GeV]")
pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")
dr_axis = hist.axis.Regular(20, 0, 5, name="dr", label=r"$\Delta R$")
eta_axis = hist.axis.Regular(20, -5, 5, name="eta", label=r"$\eta$")
phi_axis = hist.axis.Regular(50, 0, 4, name="phi", label=r"$\phi$")
nj_axis = hist.axis.Regular(10, 0, 10, name="nj", label=r"$n_{j}$")
b_axis = hist.axis.Regular(5, 0, 5, name="nb", label=r"$n_{b}$")


def makeCutSet(x, s, *args):
    return [x[s > a] for a in args]


def isGoodGenParticle(particle):
    return particle.hasFlags("isLastCopy", "fromHardProcess") & ~(
        particle.hasFlags("fromHardProcessBeforeFSR")
        & ((abs(particle.pdgId) == 1) | (abs(particle.pdgId) == 3))
    )


MCCampaign = "UL2018"
if MCCampaign == "UL2016preVFP":
    b_tag_wps = [0.0508, 0.2598, 0.6502]
elif MCCampaign == "UL2016postVFP":
    b_tag_wps = [0.0480, 0.2489, 0.6377]
elif MCCampaign == "UL2017":
    b_tag_wps = [0.0532, 0.3040, 0.7476]
elif MCCampaign == "UL2018":
    b_tag_wps = [0.0490, 0.2783, 0.7100]


def createObjects(events):
    good_jets = events.Jet[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)]
    fat_jets = events.FatJet
    (tight_top,) = makeCutSet(fat_jets, fat_jets.particleNet_TvsQCD, 0.97)
    loose_b, med_b = makeCutSet(
        good_jets, good_jets.btagDeepFlavB, b_tag_wps[0], b_tag_wps[1]
    )
    el = events.Electron
    good_electrons = el[
        (el.cutBased == 4)
        & (el.miniPFRelIso_all < 0.1)
        & (el.pt > 30)
        & (abs(el.eta) < 2.4)
    ]
    mu = events.Muon
    good_muons = mu[
        (mu.mediumId) & (mu.miniPFRelIso_all < 0.2) & (mu.pt > 30) & (abs(mu.eta) < 2.4)
    ]
    events["good_jets"] = good_jets
    events["good_electrons"] = good_electrons
    events["good_muons"] = good_muons
    events["loose_bs"] = loose_b
    events["med_bs"] = med_b
    events["tight_tops"] = tight_top
    return events, {}


def goodGenParticles(events):
    events["good_gen_particles"] = events.GenPart[isGoodGenParticle(events.GenPart)]
    gg = events.good_gen_particles
    top = gg[abs(gg.pdgId) == 1000006]
    return events


def createSelection(events):
    good_jets = events.good_jets
    fat_jets = events.FatJet

    good_muons = events.good_muons
    good_electrons = events.good_electrons

    loose_b = events.loose_bs
    tight_top = events.tight_tops

    selection = PackedSelection()

    filled_jets = ak.pad_none(good_jets, 2, axis=1)
    # filled_jets = good_jets
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    selection.add("jets", (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 5))
    selection.add("0Lep", (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0))
    selection.add("2bjet", ak.num(loose_b) >= 2)
    selection.add("highptjet", ak.fill_none(filled_jets[:, 0].pt > 300, False))
    selection.add("jet_dr", (top_two_dr < 4) & (top_two_dr > 2))
    selection.add("0Top", ak.num(tight_top) == 0)
    return selection


def makeHistogram(axis, dataset, data):
    h = hist.Hist(dataset_axis, axis, storage="weight")
    return h.fill(dataset, data)


def createJetHistograms(events):
    ret = {}
    dataset = events.metadata["dataset"]
    gj = events.good_jets

    ret[f"h_jet_pt"] = makeHistogram(phi_axis, dataset, ak.flatten(gj.pt))
    ret[f"h_njet"] = makeHistogram(nj_axis, dataset, ak.num(gj))

    for i, j in [(0, 3), (1, 4), (0, 4)]:
        jets = gj[:, i:j].sum()
        ret[f"h_m{i}{j}_pt"] = makeHistogram(pt_axis, dataset, jets.pt)
        ret[f"h_m{i}{j}_eta"] = makeHistogram(eta_axis, dataset, jets.eta)
        ret[f"h_m{i}{j}_m"] = makeHistogram(eta_axis, dataset, jets.mass)
    for i in range(0, 4):
        ret[f"h_pt_{i}"] = makeHistogram(pt_axis, dataset, gj[:, i].pt)
        ret[f"h_eta_{i}"] = makeHistogram(eta_axis, dataset, gj[:, i].eta)
        ret[f"h_phi_{i}"] = makeHistogram(phi_axis, dataset, gj[:, i].phi)

    for i, j in list(x for x in itertools.combinations(range(0, 4), 2) if x[0] != x[1]):
        pass
    return ret


def createBHistograms(events):
    ret = {}
    dataset = events.metadata["dataset"]
    l_bjets = events.loose_bs
    m_bjets = events.med_bs
    ret[f"h_loose_bjet_pt"] = makeHistogram(pt_axis, dataset, ak.flatten(l_bjets.pt))
    ret[f"h_loose_nb"] = makeHistogram(b_axis, dataset, ak.num(l_bjets.pt))
    ret[f"h_loose_bdr"] = makeHistogram(
        b_axis, dataset, l_bjets[:, 0].delta_r(l_bjets[:, 1])
    )
    return ret


def signalGenLevel(events):
    pass


def run(events):
    events, accum = createObjects(events)
    selection = createSelection(events)
    events = events[selection.all(*selection.names)]

    good_jets = events.good_jets
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    fat_jets = events.FatJet

    loose_top, med_top, tight_top = makeCutSet(
        fat_jets, fat_jets.particleNet_TvsQCD, 0.58, 0.80, 0.97
    )
    loose_W, med_W, tight_W = makeCutSet(
        fat_jets, fat_jets.particleNet_WvsQCD, 0.7, 0.94, 0.98
    )

    deep_top_wp1, deep_top_wp2, deep_top_wp3, deep_top_wp4 = makeCutSet(
        fat_jets, fat_jets.deepTag_TvsQCD, 0.436, 0.802, 0.922, 0.989
    )
    deep_W_wp1, deep_W_wp2, deep_W_wp3, deep_W_wp4 = makeCutSet(
        fat_jets, fat_jets.deepTag_WvsQCD, 0.458, 0.762, 0.918, 0.961
    )
    loose_b, med_b, tight_b = makeCutSet(
        good_jets, good_jets.btagDeepFlavB, *(b_tag_wps[x] for x in range(3))
    )

    njets = ak.num(good_jets)
    n_loose_top, n_med_top, n_tight_top = (
        ak.num(loose_top),
        ak.num(med_top),
        ak.num(tight_top),
    )
    n_loose_W, n_med_W, n_tight_W = ak.num(loose_W), ak.num(med_W), ak.num(tight_W)

    ht = ak.sum(good_jets.pt, axis=1)
    ret = createJetHistograms(events)
    ret = createBHistograms(events)
    print(ret)


class RPVProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def save_skim(self, events):
        filename = (
           "__".join(
               [
                   events.metadata['dataset'],
                   events.metadata['fileuuid'],
                   str(events.metadata['entrystart']),
                   str(events.metadata['entrystop']),
               ]
           )
           + ".root"
        )
        path = Path("skims")
        if not path.is_dir():
            path.mkdir()
        outpath = path / filename
        with uproot.recreate(outpath) as fout:
            fout["Events"] = uproot_writeable(events)

    def process(self, events):
        # print(events.metadata)
        # self.save_skim(events)
        # return [events.metadata]
        events, accum = createObjects(events)
        selection = createSelection(events)
        events = events[selection.all(*selection.names)]

        jet_hists = createJetHistograms(events)
        b_hists = createBHistograms(events)

        return accumulate([jet_hists, b_hists])

    def postprocess(self, accumulator):
        pass


if __name__ == "__main__":
    # cluster = HTCondorCluster(cores=2,memory="2GB",disk="4GB", log_directory="logs",silence_logs='debug')
    fbase = Path("samples")
    samples = [
        ("QCD2018", "QCDBEnriched2018.txt"),
        # ("Diboson2018", "Diboson2018.txt"),
        # ("WQQ2018", "WJetsToQQ2018.txt"),
        # ("ZQQ2018", "ZJetsToQQ2018.txt"),
        # ("ZNuNu2018", "ZJetsToNuNu2018.txt"),
    ]
    filesets = {
        sample: [
            f"root://cmsxrootd.fnal.gov//{f.strip()}" for f in open(fbase / fname)
        ][0:1]
        for sample, fname in samples
    }
    # events = NanoEventsFactory.from_root(
    ##            "/uscms_data/d3/ckapsiak/Analysis/SingleStop/Skim/CMSSW_10_6_19_patch2/src/PhysicsTools/NanoAODTools/python/postprocessing/SingleStop/ScaledSkims/Diboson2018.root",
    #            "skimmed.root",
    #                schemaclass=NanoAODSchema.v6,
    #                    metadata={"dataset": "DYJets"},
    #                    ).events()
    #p = RPVProcessor()
    #x = p.process(events)
    #client = Client()
    #executor=processor.DaskExecutor(client=client)
    executor = processor.FuturesExecutor(compression=None, workers=4)
    run = processor.Runner(
       executor=executor,
       schema=NanoAODSchema,
       chunksize=1000000,
    )
    output = run(filesets, "Events", processor_instance=RPVProcessor())
    with open("output.pkl", "wb") as f:
        pickle.dump(output, f)
