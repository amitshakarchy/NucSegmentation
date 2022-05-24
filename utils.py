import pandas as pd


def get_tracks(csv_path):
    """
    The method loads the tracks' csv.
    :param csv_path:
    :return:
        df: dataframes of all the cells' tracks
        all_tracks_df: list, contains a dataframe of each manually tracked cell
    """
    all_tracks_df = pd.read_csv(csv_path, encoding="cp1252")
    all_tracks_df = all_tracks_df.drop(labels=range(0, 2), axis=0)  # drop redundant rows
    all_tracks_df = all_tracks_df.astype(float)
    all_tracks_df['target'] = 1
    all_tracks_df = all_tracks_df[all_tracks_df["manual"] == 1]  # keep only manually tracked cells

    all_tracks_df.rename(columns={"Spot position": "Spot position X (µm)",
                                  "Spot position.1": "Spot position Y (µm)"},
                         inplace=True)

    # create a list of dataframes for each single track
    tracks = [group for _, group in all_tracks_df.groupby('Spot track ID')]

    return all_tracks_df, tracks


if __name__ == '__main__':
    csv_path = r"C:\Users\Amit\PycharmProjects\muscle-formation-regeneration\data\mastodon\all_detections_s3-vertices.csv"
    df, track_list = get_tracks(csv_path)
    print(df)
