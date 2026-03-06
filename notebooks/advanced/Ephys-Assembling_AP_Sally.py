
# %% [markdown]
#
# # Assemble Neuropixels data
#
# requirements:
# ```
# pip install open-ephys-python-tools
# ```
# ## 1) 
# Run:
# ```
# python -m physion.assembling.dataset build-DataTable %USERPROFILE%\DATA\2026_02_13
# ```
# this will create a file: `%USERPROFILE%\DATA\DataTable0.xlsx`  
#      move it to  ~/DATA/2026_02_13/DataTable0.xlsx
#
# Then fill its neuropixels folder (`Npx-Folder`) and recordings information (`Npx-Rec`).    
#
#       N.B. you can use the code below to guide filling the recordings info

# %%
import os
import sys, time
sys.path += [os.path.expanduser('~/physion/src'), '../../src']
import json
import numpy as np
import pandas as pd

from open_ephys.analysis import Session

from physion.assembling.dataset import read_spreadsheet
from physion.acquisition.tools import find_line_props
import physion.utils.plot_tools as pt
pt.set_style('dark')

datafolder = os.path.expanduser('~/DATA/Sally/Npx_WT_prelim_2026/2026_02_13').replace('/', os.path.sep)

INTERPROTOCOL_WINDOW = 10. # 
PROBE_NAME = 'ProbeA'
NODE = 0 # change if you have several record nodes and you want to consider another one
EXP = 1 # 


# %% [markdown]
#
# ## Load Table data

# %%
# datafolder = os.path.expanduser('~/DATA/2026_02_20').replace('/', os.path.sep)

datatable, _, analysis = read_spreadsheet(\
                        os.path.join(datafolder, 'DataTable0.xlsx'),
                                   get_metadata_from='files')
#datatable

# %% [markdown]
#
# ## Load NIdaq data

# %%
#
def load_nidaq_synch_signal(folder):
    """ """
    with open(os.path.join(folder, 'metadata.json')) as f:
        metadata = json.load(f)
    NIdaq = np.load(os.path.join(folder, 'NIdaq.npy'),
                    allow_pickle=True).item()
    props = find_line_props(
                metadata['NIdaq']['digital-outputs']['line-labels'])
    ephysSynch_signal = NIdaq['digital'][props['chan']]
    t = np.arange(len(ephysSynch_signal))*NIdaq['dt']
    pulse_onsets = t[:-1][np.flatnonzero(ephysSynch_signal[1:]>ephysSynch_signal[:-1])]
    return t, ephysSynch_signal, pulse_onsets

DF = pd.DataFrame(columns=['time', 'Npx-Rec', 'daq-nEpisodes', 'ephys-nEpisodes', 'i0', 'i1', 'nStart', 'nStop'])
DF['time'] = datatable['time']

# loop over protocols
# print(' ==== PROTOCOLS FROM NIDAQ DATA ====  ')
for iRec, protocol in enumerate(datatable['protocol']):
    _, _, onsets = load_nidaq_synch_signal(
                                os.path.join(datafolder, datatable['time'][iRec]))
    # print(' rec #%i) n=%i episodes, %s' % (iRec+1, len(onsets), protocol))
    DF.loc[iRec, 'daq-nEpisodes'] = len(onsets)

# %% [markdown]
#
# ## Load Open-Ephys data


# %%

session = Session(os.path.join(datafolder, 
                               datatable['Npx-Folder'][0]))

def build_ttl_from_events(State, Sample):
    # we start at 0
    SN, TTL = [Sample[0]-30000], [0]
    # loop over events
    for state, sample in zip(State, Sample):
        if state==1:
            SN.append(sample); TTL.append(0)
            SN.append(sample); TTL.append(1)
        if state==0:
            SN.append(sample); TTL.append(1)
            SN.append(sample); TTL.append(0)
    # we force ending at 0
    SN.append(sample); TTL.append(0)
    SN.append(sample+30000); TTL.append(0)
    return np.array(SN, dtype=np.int32), np.array(TTL, dtype=np.uint8)

def load_OpenEphys(rec):

    # find TTL events on Probe A
    cond = (rec.events['stream_name']==PROBE_NAME)

    # load the events
    State = np.array(rec.events['state'][cond])
    Sample = np.array(rec.events['sample_number'][cond])
    pulse_onsets = Sample[State==1]

    # build the time array from the set of events
    SN, TTL = build_ttl_from_events(State, Sample)
    return pulse_onsets, SN, TTL 

print(' ==== PROTOCOLS FROM OPEN-EPHYS DATA ====  ')
props = []
iRec = 0
for r, rec in enumerate(session.recordnodes[NODE].recordings):

    pulse_onsets, SN, TTL = load_OpenEphys(rec)

    fig, ax = pt.figure(axes=(1,2), ax_scale=(2.5, 1.5), hspace=0)
    fig.suptitle('Recording #%i' % (r+1))
    ax[1].set_xlabel('N, sample number (Npx Probe)')
    ax[0].set_ylabel('TTL (all)'); ax[1].set_ylabel('splitted')
    pt.plot(SN, TTL, ax=ax[0])

    # tracking different protocols
    # --> more than 2s between protocols to identify protocol changes
    iStarts = np.concatenate([[0], 
                              np.flatnonzero(np.diff(SN)>(30e3*INTERPROTOCOL_WINDOW)),
                              [len(SN)]])

    for i0, i1 in zip(iStarts[:-1], iStarts[1:]):


        irange=np.arange(i0, np.min([i1+2,len(SN)]))
        pulse_cond = (pulse_onsets>=SN[irange[0]]) & (pulse_onsets<=SN[irange[-1]])
       
        ax[1].plot(SN[irange], TTL[irange], lw=0.3, color=pt.tab10(iRec%10))
        pt.annotate(ax[1], 'protocol #%i'%(1+iRec) +iRec*'\n', (1,0), va='bottom', color=pt.tab10(iRec%10))

        DF.loc[iRec, 'i0'] = i0
        DF.loc[iRec, 'i1'] = i1
        DF.loc[iRec, 'Npx-Rec'] = 'node%i/exp%i/rec%i' % (NODE, EXP, r+1)
        DF.loc[iRec, 'ephys-nEpisodes'] = len(pulse_onsets[pulse_cond])

        iRec += 1

    pt.set_common_xlims(ax)
DF


# %%
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import least_squares


def find_sampling_match(t, nidaq_onsets, ephys_onsets):
    """
    we find the sample numbers that match the limits of the NIdaq acquisition,
    then, the samples in [nStart, nStop]
        have the time sampling:
            np.linspace(t[0], t[-1], nStop-nStart)

    where t is the nidaq time sampling array
    """

    N0 = ephys_onsets[0]
    t0 = nidaq_onsets[0]

    nMax = np.min([len(nidaq_onsets), len(ephys_onsets)])-1

    nMax=-1 # TO REMOVE 

    dN = ephys_onsets[nMax]-N0
    dT = nidaq_onsets[nMax]-t0

    if False:
        # IN CASE YOU WANT TO MAKE A MINIMIZATION FUNCTION, BUT FOR NOW not necessary...
        def to_minimize(x):
            T = (sn-x[0])*x[1]+t0
            func = interp1d(T, ttl) 
            probe_signal = func(np.clip(t, T.min(), T.max()))
            return np.mean((probe_signal-nidaqTTL)**2)

        res = least_squares(to_minimize, [N0, F0],
                            # max_nfev=10000, method='dogbox',
                            # ftol=None, xtol=None, verbose=True,
                            bounds=[(N0-3000, 0.99*F0), (N0+3000, 1.01*F0)])
        N0, F0 = res.x
    else:
        F0 = dT/dN

    nStart = N0-int(t0/F0)
    nStop = N0+dN+int((t[-1]-dT-t0)/F0) # we add dN to limit precision loss

    return nStart, nStop

def sampling_match(iRec,
                   with_fig=False):

    t, ephysSynch_signal, ephys_onsets = load_nidaq_synch_signal(
                                os.path.join(datafolder, datatable['time'][iRec]))
    

    # reload the open-ephys data:
    node = int(DF['Npx-Rec'][iRec].split('node')[1].split('/')[0])
    rec_id = int(DF['Npx-Rec'][iRec].split('rec')[1])-1
    rec = session.recordnodes[NODE].recordings[rec_id]
    # prepared ---> load
    pulse_onsets, SN, TTL = load_OpenEphys(rec)

    # restrict to previously identified range:
    irange=np.arange(DF['i0'][iRec], np.min([DF['i1'][iRec],len(SN)]))
    pulse_cond = (pulse_onsets>=SN[irange[0]]) & (pulse_onsets<=SN[irange[-1]])

    # find the matching sample range
    nStart, nStop = find_sampling_match(t, ephys_onsets, pulse_onsets[pulse_cond])

    # we now match the time sampling in the data
    cond = (SN>=nStart) & (SN<=nStop)
    T = (SN[cond]-nStart)*(t[-1]-t[0])/(nStop-nStart)
    func = interp1d(T, TTL[cond], 
                    bounds_error=False,
                    fill_value=0)
    # we build a probe signal from the interpolation of the data
    probe_signal = func(t)

    width = 1.5
    if with_fig:

        fig, AX = pt.figure(axes=(4,2), ax_scale=(1.6,.7), top=1.5, hspace=1.6, wspace=0.3)
        fig.suptitle('protocol #%i (%i episodes)' % (iRec+1, np.sum(pulse_cond)))

        for i, t0 in enumerate([0.5, t[-1]/2+1, 3.*t[-1]/4., t[-1]]):

            pt.annotate(AX[0][i], 't=%.1fs' % t0, (0.1,1))

            # nidaq
            cond = (t>(t0-width)) & (t<(t0+width))
            AX[0][i].plot(t[cond][::10], ephysSynch_signal[cond][::10])
            pt.set_plot(AX[0][i], xlabel='NIdaq time (s)', ylabel='TTL\n(from NIdaq)' if i==0 else None)

            # open-ephys
            AX[1][i].plot(t[cond][::10], probe_signal[cond][::10])
            pt.set_plot(AX[1][i], xlabel='$F \\cdot (N- N_0) $ time (s)', ylabel='TTL\n(on Probe)' if i==0 else None)

            pt.set_common_xlims([AX[0][i], AX[1][i]])

        return nStart, nStop, fig
    else:
        return nStart, nStop

# sampling_match(1, with_fig=True)

# %%
#
for iRec, time in enumerate(datatable['time']):

    DF.loc[iRec, 'nStart'], DF.loc[iRec, 'nStop'], _ =\
            sampling_match(iRec, with_fig=True)
DF

# %%

from physion.assembling.dataset import add_to_table

for key in ['Npx-Rec', 'nStart', 'nStop']:
    add_to_table(
        os.path.join(datafolder, 'DataTable0.xlsx'),
        sheet='Recordings',
        column=key,
        data=DF[key],
        insert_at=16 if 'nS' in key else 0)

# %%
#######################################################################################################



# %%
datafolder = os.path.expanduser('~/DATA/Sally/Npx_WT_prelim_2026/2026_02_13').replace('/', os.path.sep)

datatable

class Data:
    """Convenience container for one protocol/row in DataTable0.xlsx.

    Loads:
      - NIdaq digital line for visual stimulation
      - OpenEphys continuous data (named 'LFP' here; stream depends on your OpenEphys config)
      - Optional curated AP spikes from Kilosort/Phy (keeps units labeled 'good' in cluster_group.tsv)

    Notes on timebases:
      - NIdaq timebase is in seconds (t_nidaq)
      - Probe samples are indexed in Neuropixels sample numbers
      - nStart/nStop map the protocol time window onto probe sample indices.
        Spikes are aligned to protocol time by (sample - nStart) / fs_probe.
    """

    def __init__(
        self,
        datafolder: str,
        iRec: int,
        ks_folder: str | None = None,
        ks_sample_offset: int = 0,
        vis_line: int = 3,
        probe_stream: str = "ProbeA",
    ):
        # reload spreadsheet
        datatable, _, _ = read_spreadsheet(
            os.path.join(datafolder, "DataTable0.xlsx"),
            get_metadata_from="files",
        )

        # --- NIdaq ---
        nidaq = np.load(os.path.join(datafolder, datatable["time"][iRec], "NIdaq.npy"), allow_pickle=True).item()
        self.t_nidaq = np.arange(0, len(nidaq["digital"][0])) * nidaq["dt"]
        self.visStim = nidaq["digital"][vis_line]

        # protocol sample window on probe
        self.nStart = int(datatable["nStart"][iRec])
        self.nStop = int(datatable["nStop"][iRec])
        n_samples = self.nStop - self.nStart

        # time vector for probe samples within this protocol (same mapping used elsewhere in this script)
        # Use endpoint-inclusive mapping consistent with np.linspace.
        if n_samples > 1 and float(self.t_nidaq[-1]) > 0:
            self.fs_probe = (n_samples - 1) / float(self.t_nidaq[-1])
        else:
            # fallback (Neuropixels AP is typically 30 kHz)
            self.fs_probe = 30_000.0

        self.t_probe = np.linspace(0, float(self.t_nidaq[-1]), n_samples) if n_samples > 0 else np.array([])

        # --- Open Ephys continuous data ---
        session = Session(os.path.join(datafolder, datatable["Npx-Folder"][iRec]))

        node = int(DF["Npx-Rec"][iRec].split("node")[1].split("/")[0])
        rec_id = int(DF["Npx-Rec"][iRec].split("rec")[1]) - 1
        rec = session.recordnodes[node].recordings[rec_id]

        # We load the LFP for the specified stream and restrict to the protocol window [nStart:nStop].
        self.LFP = rec.continuous['ProbeA'].samples[self.nStart:self.nStop,:]
        
        
        # We load the curated spike from each recording
        #   - spike_times.npy      (sample indices)
        #   - spike_clusters.npy   (cluster id per spike; reflects Phy merges/splits)
        #   - cluster_group.tsv    (manual labels; we keep only group == "good")

        self.spikes_sample_unit = None  # shape (N,2): [sample_index (global), unit_id]
        self.unit_ids = np.array([], dtype=np.int32)
        self.good_unit_ids = np.array([], dtype=np.int32)
        self.spike_times_sec_by_unit = {}  # unit_id -> np.ndarray of spike times in seconds (protocol timebase)

        if ks_folder is not None:
            spike_times_path = os.path.join(ks_folder, "spike_times.npy")
            spike_clusters_path = os.path.join(ks_folder, "spike_clusters.npy")
            cluster_group_path = os.path.join(ks_folder, "cluster_group.tsv")

            if not os.path.exists(spike_times_path):
                raise FileNotFoundError(f"Missing {spike_times_path} in {ks_folder}")
            if not os.path.exists(spike_clusters_path):
                raise FileNotFoundError(f"Missing {spike_clusters_path} in {ks_folder}")

            # cluster_group.tsv is written by Phy when you save manual curation.
            # If absent, fall back to KS automatic labels if available.
            if not os.path.exists(cluster_group_path):
                alt = os.path.join(ks_folder, "cluster_KSLabel.tsv")
                if os.path.exists(alt):
                    cluster_group_path = alt
                else:
                    raise FileNotFoundError(
                        f"Missing {cluster_group_path} (and no cluster_KSLabel.tsv) in {ks_folder}. "
                        "Open Phy and save your curation first."
                    )

            # Load labels (robust to slightly different column names across outputs)
            cg = pd.read_csv(cluster_group_path, sep="\t")
            if "cluster_id" not in cg.columns:
                for cand in ("id", "cluster"):
                    if cand in cg.columns:
                        cg = cg.rename(columns={cand: "cluster_id"})
                        break
            if "group" not in cg.columns:
                for cand in ("KSLabel", "label", "group_label"):
                    if cand in cg.columns:
                        cg = cg.rename(columns={cand: "group"})
                        break
            if "cluster_id" not in cg.columns or "group" not in cg.columns:
                raise ValueError(
                    f"Cannot parse cluster labels from {cluster_group_path}. "
                    f"Found columns: {list(cg.columns)}"
                )

            good_ids = cg.loc[cg["group"].astype(str) == "good", "cluster_id"].to_numpy(dtype=int)
            self.good_unit_ids = good_ids.astype(np.int32)

            # Load spikes (curated assignments)
            spike_samples = np.load(spike_times_path).squeeze().astype(np.int64)
            spike_units = np.load(spike_clusters_path).squeeze().astype(np.int32)

            if spike_samples.shape[0] != spike_units.shape[0]:
                raise ValueError(
                    f"spike_times and spike_clusters length mismatch: "
                    f"{spike_samples.shape[0]} vs {spike_units.shape[0]}"
                )

            # Optional offset if binary started later than probe sample 0
            if ks_sample_offset != 0:
                spike_samples = spike_samples + int(ks_sample_offset)

            # Keep only 'good' units (if none labeled good, result will be empty)
            if self.good_unit_ids.size > 0:
                is_good = np.isin(spike_units, self.good_unit_ids)
                spike_samples = spike_samples[is_good]
                spike_units = spike_units[is_good]
            else:
                spike_samples = np.array([], dtype=np.int64)
                spike_units = np.array([], dtype=np.int32)

            # Keep only spikes in this protocol window
            in_win = (spike_samples >= self.nStart) & (spike_samples < self.nStop)
            spike_samples = spike_samples[in_win]
            spike_units = spike_units[in_win]

            self.spikes_sample_unit = np.column_stack([spike_samples, spike_units]).astype(np.int64, copy=False)
            self.unit_ids = np.unique(spike_units)

            # Convert to protocol time (seconds)
            sample_rel = spike_samples - self.nStart
            t_spikes = sample_rel / self.fs_probe

            # Per-unit dict (sorted)
            for uid in self.unit_ids:
                cond = spike_units == uid
                self.spike_times_sec_by_unit[int(uid)] = np.sort(t_spikes[cond].astype(np.float64, copy=False))

    def get_vis_onsets(self) -> np.ndarray:
        """Visual stimulus onsets from NIdaq digital line (seconds)."""
        idx = np.flatnonzero(self.visStim[1:] > self.visStim[:-1])
        return self.t_nidaq[:-1][idx]

    def psth(
        self,
        events_sec: np.ndarray | None = None,
        t_pre: float = 0.5,
        t_post: float = 1.5,
        bin_size: float = 0.010,
        smooth_sigma_bins: float | None = None,
        unit_ids: list[int] | None = None,
    ):
        """Compute PSTH (Hz) for selected units.

        Returns:
            t_centers: (n_bins,)
            psth_hz: (n_units, n_bins)
        """
        if events_sec is None:
            events_sec = self.get_vis_onsets()
        events_sec = np.asarray(events_sec, dtype=np.float64)
        if events_sec.size == 0:
            raise ValueError("No events provided/found for PSTH.")

        if self.spikes_sample_unit is None:
            raise ValueError("No spikes loaded. Pass ks_folder=... when constructing Data.")

        if unit_ids is None:
            unit_ids = [int(u) for u in self.unit_ids]

        # bins
        edges = np.arange(-t_pre, t_post + bin_size, bin_size, dtype=np.float64)
        centers = (edges[:-1] + edges[1:]) / 2.0

        psth = np.zeros((len(unit_ids), centers.size), dtype=np.float64)

        for i, uid in enumerate(unit_ids):
            st = self.spike_times_sec_by_unit.get(int(uid), np.array([], dtype=np.float64))
            if st.size == 0:
                continue

            # accumulate hist across events
            h = np.zeros(centers.size, dtype=np.float64)
            for e in events_sec:
                rel = st - float(e)
                rel = rel[(rel >= -t_pre) & (rel <= t_post)]
                if rel.size:
                    h += np.histogram(rel, bins=edges)[0]

            # convert to rate (Hz): spikes / (n_events * bin_width)
            psth[i, :] = h / (events_sec.size * bin_size)

        if smooth_sigma_bins is not None:
            from scipy.ndimage import gaussian_filter1d
            psth = gaussian_filter1d(psth, sigma=float(smooth_sigma_bins), axis=1, mode="nearest")

        return centers, psth


# %%
# LFP and visual stim for one protocol
iRec = 0


# KS_FOLDER = os.path.expanduser("~/DATA/Sally/Npx_WT_prelim_2026/2026_02_13/AP/Rec2/kilosort4_new")

data = Data(datafolder, iRec)

t0, length = 0, 60
fig, AX = pt.figure(axes_extents=[[[1,3]],[[1,1]]], ax_scale=(3,1))

SHIFT = 1000 # 1mV between each channel
cond = (data.t_probe>t0) & (data.t_probe<(t0+length))

for chan in range(10):
    lfp = data.LFP[cond,chan]
    lfp = lfp-lfp.mean()
    AX[0].plot(data.t_probe[cond], lfp+chan*SHIFT, lw=0.5, color=pt.plt.cm.tab20(chan))
pt.set_plot(AX[0], ['bottom'], ylabel='LFP')
pt.draw_bar_scales(AX[0], Xbar=1e-3, Ybar=2000, Ybar_label='2mv')

cond = (data.t_nidaq>t0) & (data.t_nidaq<(t0+length))
AX[1].plot(data.t_nidaq[cond], data.visStim[cond])
pt.set_plot(AX[1], ['bottom'], xlabel='time (s)', ylabel='vis. stim.\n onset')

# %%
from scipy.ndimage import gaussian_filter1d
events = data.t_nidaq[np.flatnonzero(data.visStim[1:]>data.visStim[:-1])]

lfp_events = []
for e in events:
    cond = (data.t_probe>(e-1)) & (data.t_probe<(e+2))
    # lfp = gaussian_filter1d(data.LFP[cond,:].mean(axis=-1), 500)
    lfp = gaussian_filter1d(data.LFP[cond,0], 500)
    pre = (data.t_probe[cond]>(e-1)) & (data.t_probe[cond]<e)
    lfp_events.append(lfp-lfp[pre].mean())
t = data.t_probe[cond]-e

fig, ax = pt.figure(ax_scale=(2,3))
pt.plot(t, 1e-3*np.mean(lfp_events, axis=0), sy=1e-3*np.std(lfp_events, axis=0), ax=ax)
pt.set_plot(ax, xlabel='time from stim. (s)', ylabel='LFP (mV)')

# %%
datatable


# %%
# --- PSTH from curated AP spikes (if KS_FOLDER is set) ---

# --- Example usage: load one protocol + curated AP spikes and compute PSTH ---
# Set this to your Kilosort output folder (where spike_times.npy / spike_clusters.npy / cluster_group.tsv lives)

iRec = 0

KS_FOLDER = os.path.expanduser("~/DATA/Sally/Npx_WT_prelim_2026/2026_02_13/AP/Rec2/kilosort4_new")

data = Data(datafolder, iRec, ks_folder=KS_FOLDER)
# Visual stim events (seconds)
events = data.get_vis_onsets()

if KS_FOLDER is not None:
    # Per-unit PSTH matrix (Hz)
    t_psth, psth_hz = data.psth(events_sec=events, t_pre=0.5, t_post=1.5, bin_size=0.01, smooth_sigma_bins=1.0)

    # Population mean +/- SEM across units
    mean_rate = psth_hz.mean(axis=0) if psth_hz.size else np.zeros_like(t_psth)
    sem_rate = psth_hz.std(axis=0) / np.sqrt(psth_hz.shape[0]) if psth_hz.shape[0] > 1 else np.zeros_like(t_psth)

    fig, ax = pt.figure(ax_scale=(2.5, 2.2))
    pt.plot(t_psth, mean_rate, sy=sem_rate, ax=ax)
    pt.set_plot(ax, xlabel="time from stim (s)", ylabel="firing rate (Hz)", title="PSTH (good units)")

    # Save outputs next to Kilosort folder
    # np.save(os.path.join(KS_FOLDER, "psth_time_s.npy"), t_psth)
    # np.save(os.path.join(KS_FOLDER, "psth_hz_units_x_time.npy"), psth_hz)

# %%
import numpy as np
import matplotlib.pyplot as plt

resp_win = (0.02, 0.2)
base_win = (-0.2, 0.0)

resp_mask = (t_psth >= resp_win[0]) & (t_psth < resp_win[1])
base_mask = (t_psth >= base_win[0]) & (t_psth < base_win[1])

# response metric = response mean - baseline mean
score = psth_hz[:, resp_mask].mean(axis=1) - psth_hz[:, base_mask].mean(axis=1)
order = np.argsort(score)[::-1]  # descending

p_sorted = psth_hz[order]

plt.figure(figsize=(10, 6))
im = plt.imshow(
    p_sorted,
    aspect="auto",
    origin="lower",
    extent=[t_psth[0], t_psth[-1], 0, p_sorted.shape[0]],
)
plt.axvline(0, linewidth=1)
plt.colorbar(im, label="Firing rate (Hz)")
plt.xlabel("Time from stimulus onset (s)")
plt.ylabel("Unit (sorted by evoked Δrate)")
plt.title("PSTH heatmap (sorted)")
plt.tight_layout()
plt.show()



# %%
# baseline normalise
base = psth_hz[:, base_mask]
mu = base.mean(axis=1, keepdims=True)
sd = base.std(axis=1, keepdims=True) + 1e-9

p_z = (psth_hz - mu) / sd
p_z = p_z[order] # reuse sorting from above if you want

plt.figure(figsize=(10, 6))
im = plt.imshow( p_z, aspect="auto", origin="lower",
                extent=[t_psth[0], t_psth[-1], 0, p_z.shape[0]])

plt.axvline(0, linewidth=1)
plt.colorbar(im, label="Z (relative to baseline)")
plt.xlabel("Time from stimulus onset (s)")
plt.ylabel("Unit")
plt.title("PSTH heatmap (baseline z-score)")
plt.tight_layout()
plt.show()


# %%
