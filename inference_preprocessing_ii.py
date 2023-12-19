import pickle
import click
import numpy as np
import pandas as pd
from pathlib import Path

from Preprocessing_Billboard import compute_Tonal_centroids, enharmonic, chord2int


def preprocess_data_read(
    song_name: str, chroma_path: pd.DataFrame, annotations_path: pd.DataFrame
):

    annotations_df = pd.read_csv(
        annotations_path, header=None, sep="\t", names=["start", "end", "chord"]
    )

    annotations = {}
    adt = [
        ("onset", np.float32),
        ("end", np.float32),
        ("chord", object),
    ]  # dtype of annotations

    song_annotations = []
    for __, row in annotations_df.iterrows():
        annotation_triplet = (row["start"], row["end"], row["chord"])
        song_annotations.append(annotation_triplet)
    annotations[song_name] = np.array(song_annotations, dtype=adt)

    inference_data = {}
    dt = [
        ("op", object),
        ("onset", np.float32),
        ("chroma", object),
        ("chord", np.int32),
        ("chordChange", np.int32),
    ]  # dtype of output data

    rows = np.genfromtxt(chroma_path, delimiter=",")
    frames = []
    pre_chord = None
    for r, row in enumerate(rows):
        onset = row[0]
        chroma1 = row[1:13]
        chroma2 = row[13:25]
        chroma1_norm = chroma1
        chroma2_norm = chroma2
        both_chroma = np.concatenate([chroma1_norm, chroma2_norm]).astype(np.float32)

        label = annotations[song_name][
            (annotations[song_name]["onset"] <= onset) & (annotations[song_name]["end"] > onset)
        ]
        try:
            chord = label["chord"][0]
            root = chord.split(":")[0]
            if "b" in root:
                chord = enharmonic(chord)
            chord_int = chord2int(chord)
        except:
            print("ErrorMessage: cannot find label: piece %s, onset %f" % (song_name, onset))
            quit()
        chordChange = 0 if chord_int == pre_chord else 1
        pre_chord = chord_int

        frames.append((song_name, onset, both_chroma, chord_int, chordChange))

    inference_data[song_name] = np.array(frames, dtype=dt)  # [time, ]

    """ BillboardData = {'Name': structured array with fileds = ('op', 'onset', 'chroma',  'chord', 'chordChange'), ...} """
    # keys: ['0001', '0002', ...]
    # values: structured array with frames = ('op', 'onset', 'chroma',  'chord', 'chordChange')

    return inference_data

def preprocess_data_augment(inference_data: dict):

    shift = 0

    def shift_chromagram(chromagram, shift):
        if shift > 0:
            chr1 = np.roll(chromagram[:, :12], shift, axis=1)
            chr2 = np.roll(chromagram[:, 12:], shift, axis=1)
            chromagram = np.concatenate([chr1, chr2], axis=1)
        return chromagram

    def shif_chord(chord, shift):
        if chord < 12:
            new_chord = (chord + shift) % 12
        elif chord < 24:
            new_chord = (chord - 12 + shift) % 12 + 12
        else:
            new_chord = chord
        return new_chord

    """ inference_data = {'Name': structured array with fileds = ('op', 'onset', 'chroma',  'chord', 'chordChange'), ...} """
    """inference_data_augment = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array}, ...}"""
    inference_data_augment = {}
    for key, value in inference_data.items():
        chromagram = np.array([x for x in value["chroma"]])
        chord = value["chord"]
        chordChange = value["chordChange"]

        chromagram_shift = shift_chromagram(chromagram, shift)
        TC_shift = compute_Tonal_centroids(
            (chromagram_shift[:, :12] + chromagram_shift[:, 12:]) / 2
        )  # [time, 6]
        chord_shift = np.array([shif_chord(x, shift) for x in chord])
        chordChange_shift = chordChange

        inference_data_augment[key] = {}
        inference_data_augment[key]["chroma"] = chromagram_shift
        inference_data_augment[key]["TC"] = TC_shift
        inference_data_augment[key]["chord"] = chord_shift
        inference_data_augment[key]["chordChange"] = chordChange_shift

    return inference_data_augment

def preprocess_data_segment(inference_data_augment: dict, segment_width=21, segment_hop=5):
    print("Running Message: segment data ...")

    # conversion of frames based representation to segments
    # each segment has a length of segment_width = 21 ~ 1 frame = 0.046 sec, segment ~ 0.5 sec
    # the hop size is segment_hop = 5 ~ 0.25 sec for overlapping

    inference_data_segment = {}

    for key, value in inference_data_augment.items():
        chroma = value["chroma"]  # [time, 24]
        TC = value["TC"]  # [time, 6]
        chroma_TC = np.concatenate([chroma, TC], axis=1)  # [time, 30]
        del chroma, TC
        chord = value["chord"]  # [time,]

        n_pad = segment_width // 2
        chroma_TC_pad = np.pad(
            chroma_TC, ((n_pad, n_pad), (0, 0)), "constant", constant_values=0.0
        )  # [time + 2*n_pad, 30]
        chord_pad = np.pad(
            chord, (n_pad, n_pad), "constant", constant_values=24
        )  # [time + 2*n_pad,]

        n_frames = chroma_TC.shape[0]
        chroma_TC_segment = np.array(
            [
                chroma_TC_pad[i - n_pad : i + n_pad + 1]
                for i in range(n_pad, n_pad + n_frames, segment_hop)
            ]
        )  # [n_segments, segment_width, 30]
        chroma_segment = np.reshape(
            chroma_TC_segment[:, :, :24], [-1, segment_width * 24]
        )  # [n_segments, segment_widt*24]
        TC_segment = np.reshape(
            chroma_TC_segment[:, :, 24:], [-1, segment_width * 6]
        )  # [n_segments, segment_widt*6]
        chord_segment = np.array(
            [chord_pad[i] for i in range(n_pad, n_pad + n_frames, segment_hop)]
        )  # [n_segments,]
        chordChange_segment = np.array(
            [1]
            + [
                0 if x == y else 1
                for x, y in zip(chord_segment[1:], chord_segment[:-1])
            ]
        )
        del chroma_TC_segment

        """BillboardData_segment = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array}, ...}"""
        inference_data_segment[key] = {}
        inference_data_segment[key]["chroma"] = chroma_segment.astype(
            np.float32
        )
        inference_data_segment[key]["TC"] = TC_segment.astype(np.float32)
        inference_data_segment[key]["chord"] = chord_segment.astype(np.int32)
        inference_data_segment[key][
            "chordChange"
        ] = chordChange_segment.astype(np.int32)

    return inference_data_segment

def preprocess_data_reshape(inference_data_segment: dict, out_dir: Path, n_steps=100):
    print("Running Message: reshape inference data segment ...")

    inference_data_reshape = {}
    for key, value in inference_data_segment.items():
        chroma = value["chroma"]
        TC = value["TC"]
        chord = value["chord"]
        chordChange = value["chordChange"]

        n_frames = chroma.shape[0]
        n_pad = 0 if n_frames / n_steps == 0 else n_steps - (n_frames % n_steps)
        if n_pad != 0:  # chek if need paddings
            chroma = np.pad(
                chroma, [(0, n_pad), (0, 0)], "constant", constant_values=0
            )
            TC = np.pad(TC, [(0, n_pad), (0, 0)], "constant", constant_values=0)
            chord = np.pad(
                chord, [(0, n_pad)], "constant", constant_values=24
            )  # 24 for padding frams
            chordChange = np.pad(
                chordChange, [(0, n_pad)], "constant", constant_values=0
            )  # 0 for padding frames

        seq_hop = n_steps // 2
        n_sequences = int((chroma.shape[0] - n_steps) / seq_hop) + 1
        _, feature_size = chroma.shape
        _, TC_size = TC.shape
        s0, s1 = chroma.strides
        chroma_reshape = np.lib.stride_tricks.as_strided(
            chroma,
            shape=(n_sequences, n_steps, feature_size),
            strides=(s0 * seq_hop, s0, s1),
        )
        ss0, ss1 = TC.strides
        TC_reshape = np.lib.stride_tricks.as_strided(
            TC,
            shape=(n_sequences, n_steps, TC_size),
            strides=(ss0 * seq_hop, ss0, ss1),
        )
        (sss0,) = chord.strides
        chord_reshape = np.lib.stride_tricks.as_strided(
            chord, shape=(n_sequences, n_steps), strides=(sss0 * seq_hop, sss0)
        )
        (ssss0,) = chordChange.strides
        chordChange_reshape = np.lib.stride_tricks.as_strided(
            chordChange,
            shape=(n_sequences, n_steps),
            strides=(ssss0 * seq_hop, ssss0),
        )
        sequenceLen = np.array(
            [n_steps for _ in range(n_sequences - 1)] + [n_steps - n_pad],
            dtype=np.int32,
        )  # [n_sequences]

        # reshape data into two dimensional data
        # original shape e.g. chroma.shape = (700, 504)
        # now shape e.g. chroma_reshape.shape = (13, 100, 504) for n_steps = 100

        """inference_data_reshape = {'op': {'chroma': array, 'TC': array, 'chord': array, 'chordChange': array, 'sequenceLen': array, 'nSequence': array}, ...}"""
        inference_data_reshape[key] = {}
        inference_data_reshape[key]["chroma"] = chroma_reshape
        inference_data_reshape[key]["TC"] = TC_reshape
        inference_data_reshape[key]["chord"] = chord_reshape
        inference_data_reshape[key]["chordChange"] = chordChange_reshape
        inference_data_reshape[key]["sequenceLen"] = sequenceLen
        inference_data_reshape[key]["nSequence"] = n_sequences
        # inference_data_reshape[key]['nSegment'] = (n_sequences // 2 + 1)*n_steps - n_pad
        
        with open(out_dir / "final_reshaped_data.pkl", "wb") as f:
            pickle.dump(inference_data_reshape, f)
        print(f"reshaped inference data saved at {out_dir}")



@click.command()
@click.option("--song_name", type=str, default="penny_lane", help="Song name")
@click.option(
    "--chroma_path",
    type=Path,
    default="own_examples/penny_lane/penny_lane_bothchroma_transformed.csv",
    help="Chroma path",
)
@click.option(
    "--annotations_path",
    type=Path,
    default="own_examples/penny_lane/penny_lane_chord_estimate_transformed.csv",
    help="Annotations path",
)
def main(song_name: str, chroma_path: Path, annotations_path: Path):

    inference_data = preprocess_data_read(
        song_name=song_name, chroma_path=chroma_path, annotations_path=annotations_path
    )
    inference_data_augment = preprocess_data_augment(inference_data=inference_data)
    infererence_data_segment = preprocess_data_segment(inference_data_augment=inference_data_augment)

    preprocess_data_reshape(inference_data_segment=infererence_data_segment, out_dir=chroma_path.parent)


if __name__ == "__main__":
    main()
