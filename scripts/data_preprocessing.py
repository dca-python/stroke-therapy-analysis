import pandas as pd
import numpy as np

def clean(path):
    """Takes the path of raw data, selects a sheet, selects columns, constructs variables,
    homogenizes missing values, extrapolates missing values, transforms the data in long format,
    translates variables to english and returns a pandas dataframe"""
    variables = [
        "Probandennr", "Geschlecht", "Haendigkeit", "Hemiseite", "Alter_Aufnahme_Reha", "Alter_ET",
        "Diagnose_auswahl", "Diagnose_3", "AB3_1", "AB3_2", "AB3_3", "AB3_4", "AB4_1", "AB4_2", "AB4_3",
        "AB4_4", "AB4_5", "AB5_1", "AB5_2", "AB5_3", "AB5_4", "AB5_5", "AB6_1", "AB6_2", "AB6_3", "AB6_4",
        "AB7_1", "AB7_2", "AB7_3", "AB7_4", "AB8_1", "AB8_2", "AB8_3", "AB8_4", "AB9_1", "AB9_2", "AB9_3",
        "AB9_4", "AB9_5", "AB10_1", "AB10_2", "AB10_3", "AB11_1", "AB11_2", "AB11_3", "AB12_1", "AB12_2",
        "AB12_3", "AB13_1", "AB13_2", "AB13_3", "AB13_4", "AB14_1", "AB14_2", "AB14_3", "AB15_1", "AB15_2",
        "AB15_3", "AB16_1", "AB16_2", "AB16_3", "AB17_1", "AB17_2", "AB17_3", "AB18_1", "AB18_2", "AB18_3",
        "AB19_1", "AB19_2", "AB19_3", "AB20_1", "AB20_2", "AB20_3", "AB21_1", "AB21_2", "AB22_1", "AB22_2",
        "AB23_1", "AB23_2", "AB23_3", "AB24_1", "AB24_2", "AB25_1", "AB25_2", "AB25_3", "AB26_1", "AB26_2",
        "AB27_1", "AB27_2", "AB27_3", "AB28_1", "AB28_2", "AB29_1", "AB29_2", "AB29_3", "AB30_1", "AB30_2",
        "AS3_1", "AS3_2", "AS3_3", "AS3_4", "AS4_1", "AS4_2", "AS4_3", "AS4_4", "AS4_5", "AS5_1", "AS5_2",
        "AS5_3", "AS5_4", "AS5_5", "AS6_1", "AS6_2", "AS6_3", "AS6_4", "AS7_1", "AS7_2", "AS7_3", "AS7_4",
        "AS8_1", "AS8_2", "AS8_3", "AS8_4", "AS9_1", "AS9_2", "AS9_3", "AS9_4", "AS9_5", "AS10_1", "AS10_2",
        "AS10_3", "AS11_1", "AS11_2", "AS11_3", "AS12_1", "AS12_2", "AS12_3", "AS13_1", "AS13_2", "AS13_3",
        "AS13_4", "AS14_1", "AS14_2", "AS14_3", "AS15_1", "AS15_2", "AS15_3", "AS16_1", "AS16_2", "AS16_3",
        "AS17_1", "AS17_2", "AS17_3", "AS18_1", "AS18_2", "AS18_3", "AS19_1", "AS19_2", "AS19_3", "AS20_1",
        "AS20_2", "AS20_3", "AS21_1", "AS21_2", "AS22_1", "AS22_2", "AS23_1", "AS23_2", "AS23_3", "AS24_1",
        "AS24_2", "AS25_1", "AS25_2", "AS25_3", "AS26_1", "AS26_2", "AS27_1", "AS27_2", "AS27_3", "AS28_1",
        "AS28_2", "AS29_1", "AS29_2", "AS29_3", "AS30_1", "AS30_2", "FuglMeyer_Ges_Erfolg"
    ]
    attention_mapping = {
        'konzentriert': 3,
        "angestrengt konzentriert": 2,
        "kurzzeitig konzentriert": 1,
        "abwesend": 0,
        0: pd.NA
    }
    data = pd.read_excel(
        io=path,
        sheet_name="reduzierter Datensatz"  # the sheet containing the raw data
    )
    data_clean = (data
        [variables]
        .assign(
                weiblich=lambda df_: df_.Geschlecht.map({"männlich": 0, "weiblich": 1}),
                rechtshändig=lambda df_: df_.Haendigkeit.map({"links": 0, "rechts": 1}),
                links_betroffen=lambda df_: df_.Hemiseite.map({"rechts": 0, "links": 1}),
                diagnose_infarkt=lambda df_: df_.Diagnose_3.map({"Blutung": 0, "Infarkt": 1}),
                erstmaliger_schlaganfall=lambda df_: df_.Diagnose_auswahl.map({
                    '"mehrmaliger Schlaganfall"': 0, '"erstmaliger Schlaganfall"': 1
                    }),
                fugl_meyer_gesamt_erfolg=lambda df_: df_.FuglMeyer_Ges_Erfolg.map({
                    'kein Erfolg <=2': 0, 'Erfolg >=2': 1
                    }),
    )
    .replace(999, 0)  # 999 for patient 52, 53 in AB18_1; extrapolated with a zero for non-participation
    .fillna(1) # NA for patient 69 in column erstmaliger_schlaganfall; extrapolated with majority (94,92 %) value 1
    .drop(columns=["Geschlecht", "Haendigkeit", "Hemiseite", "Diagnose_3",
                   "Diagnose_auswahl", "FuglMeyer_Ges_Erfolg", "Alter_ET"])
    )
    session_cols = [
        'AB3_1', 'AB3_2', 'AB3_3', 'AB3_4', 'AB4_1', 'AB4_2', 'AB4_3', 'AB4_4', 'AB4_5', 'AB5_1', 'AB5_2',
        'AB5_3', 'AB5_4', 'AB5_5', 'AB6_1', 'AB6_2', 'AB6_3', 'AB6_4', 'AB7_1', 'AB7_2', 'AB7_3', 'AB7_4',
        'AB8_1', 'AB8_2', 'AB8_3', 'AB8_4', 'AB9_1', 'AB9_2', 'AB9_3', 'AB9_4', 'AB9_5', 'AB10_1', 'AB10_2',
        'AB10_3', 'AB11_1', 'AB11_2', 'AB11_3', 'AB12_1', 'AB12_2', 'AB12_3', 'AB13_1', 'AB13_2', 'AB13_3',
        'AB13_4', 'AB14_1', 'AB14_2', 'AB14_3', 'AB15_1', 'AB15_2', 'AB15_3', 'AB16_1', 'AB16_2', 'AB16_3',
        'AB17_1', 'AB17_2', 'AB17_3', 'AB18_1', 'AB18_2', 'AB18_3', 'AB19_1', 'AB19_2', 'AB19_3', 'AB20_1',
        'AB20_2', 'AB20_3', 'AB21_1', 'AB21_2', 'AB22_1', 'AB22_2', 'AB23_1', 'AB23_2', 'AB23_3', 'AB24_1',
        'AB24_2', 'AB25_1', 'AB25_2', 'AB25_3', 'AB26_1', 'AB26_2', 'AB27_1', 'AB27_2', 'AB27_3', 'AB28_1',
        'AB28_2', 'AB29_1', 'AB29_2', 'AB29_3', 'AB30_1', 'AB30_2', 'AS3_1', 'AS3_2', 'AS3_3', 'AS3_4',
        'AS4_1', 'AS4_2', 'AS4_3', 'AS4_4', 'AS4_5', 'AS5_1', 'AS5_2', 'AS5_3', 'AS5_4', 'AS5_5', 'AS6_1',
        'AS6_2', 'AS6_3', 'AS6_4', 'AS7_1', 'AS7_2', 'AS7_3', 'AS7_4', 'AS8_1', 'AS8_2', 'AS8_3', 'AS8_4',
        'AS9_1', 'AS9_2', 'AS9_3', 'AS9_4', 'AS9_5', 'AS10_1', 'AS10_2', 'AS10_3', 'AS11_1', 'AS11_2',
        'AS11_3', 'AS12_1', 'AS12_2', 'AS12_3', 'AS13_1', 'AS13_2', 'AS13_3', 'AS13_4', 'AS14_1', 'AS14_2',
        'AS14_3', 'AS15_1', 'AS15_2', 'AS15_3', 'AS16_1', 'AS16_2', 'AS16_3', 'AS17_1', 'AS17_2', 'AS17_3',
        'AS18_1', 'AS18_2', 'AS18_3', 'AS19_1', 'AS19_2', 'AS19_3', 'AS20_1', 'AS20_2', 'AS20_3', 'AS21_1',
        'AS21_2', 'AS22_1', 'AS22_2', 'AS23_1', 'AS23_2', 'AS23_3', 'AS24_1', 'AS24_2', 'AS25_1', 'AS25_2',
        'AS25_3', 'AS26_1', 'AS26_2', 'AS27_1', 'AS27_2', 'AS27_3', 'AS28_1', 'AS28_2', 'AS29_1', 'AS29_2',
        'AS29_3', 'AS30_1', 'AS30_2'
        ]
    data_clean[session_cols] = data_clean[session_cols].replace(attention_mapping)
    id_variables = ['Probandennr', 'Alter_Aufnahme_Reha', 'weiblich', 'rechtshändig', 'links_betroffen',
                    'diagnose_infarkt', 'erstmaliger_schlaganfall', 'fugl_meyer_gesamt_erfolg']
    data_clean = (pd.melt(
        data_clean,
        id_vars=id_variables,
        var_name="aufmerksamkeitsform_sitzung_phase",
        value_name='aufmerksamkeit')
    )
    data_clean[['aufmerksamkeitstyp', 'sitzung_phase']] = (data_clean
            ['aufmerksamkeitsform_sitzung_phase']
            .str.extract(r'([A-Z]+)([0-9_]+)')
        )
    data_clean[['sitzung', 'phase']] = (data_clean
            ['sitzung_phase']
            .str.extract(r'(\d+)_(\d+)')
        )
    data_clean.drop(columns=['sitzung_phase'], inplace=True)
    data_clean['sitzung'] = data_clean['sitzung'].astype(int)
    data_clean['phase'] = data_clean['phase'].astype(int)
    data_clean = data_clean.dropna()
    data_clean['aufmerksamkeit'] = data_clean['aufmerksamkeit'].astype(int)
    data_clean['erstmaliger_schlaganfall'] = data_clean['erstmaliger_schlaganfall'].astype(int)
    english_column_mapping = {
        'Probandennr': 'participant_id',
        'Alter_Aufnahme_Reha': 'age_rehab_admission',
        'weiblich': 'female',
        'rechtshändig': 'right_handed',
        'links_betroffen': 'left_affected',
        'diagnose_infarkt': 'diagnosis_infarction',
        'erstmaliger_schlaganfall': 'first_stroke',
        'fugl_meyer_gesamt_erfolg': 'therapy_success',
        'aufmerksamkeitsform_sitzung_phase': 'attention_form_session_phase',
        'aufmerksamkeit': 'attention_score',
        'aufmerksamkeitstyp': 'attention_type',
        'sitzung': 'session',
        'phase': 'phase'
    }
    data_clean = data_clean.rename(columns=english_column_mapping)
    return data_clean

def fill_missing_with_grouped_mean(df, group_id_cols: list, mean_value_cols: list):
    """Fills out missing values in certain mean_value_cols by their group mean,
    where the grouping variables are defined by group_id_cols."""
    for mean_value_col in mean_value_cols:
        mean_values = df.groupby(group_id_cols)[mean_value_col].transform('mean')
        df[mean_value_col] = df[mean_value_col].fillna(mean_values)
    return df

def seperate_attention_score_variables(clean_data):
    """Takes the cleaned dataframe from clean() and redefines the attention
    variables to have seperate columns for both attention types.
    Fills out missing values in certain mean_value_cols by their group mean,
    where the grouping variables are defined by group_id_cols."""
    df = clean_data
    attention_score_ab = (df
        .assign(attention_score_ab=np.where(df.attention_type == 'AB', df.attention_score, pd.NA))
        .assign(attention_score_as=np.where(df.attention_type == 'AS', df.attention_score, pd.NA))
        [["participant_id", "session", "phase","attention_score_ab"]]
        .dropna()
    )
    attention_score_as = (df
        .assign(attention_score_ab=np.where(df.attention_type == 'AB', df.attention_score, pd.NA))
        .assign(attention_score_as=np.where(df.attention_type == 'AS', df.attention_score, pd.NA))
        [["participant_id", "session", "phase","attention_score_as"]]
        .dropna()
    )
    attention_df = pd.merge(left=attention_score_ab, right=attention_score_as)
    attention_free_df = (df
        .drop(columns=["attention_form_session_phase", "attention_score", "attention_type"])
    )
    attention_type_split_df = (
        pd.merge(attention_free_df, attention_df, how="left", on=["participant_id", "session", "phase"])
        .assign(as_minus_ab=lambda df_: df_.attention_score_as - df_.attention_score_ab)
        .reset_index()
        .drop(columns=["index"])
        .drop_duplicates()
    )
    attention_type_split_clean_df = (fill_missing_with_grouped_mean( # After extensively examining missing values
        df=attention_type_split_df,
        group_id_cols=["participant_id"],
        mean_value_cols=["attention_score_ab", "attention_score_as", "as_minus_ab"])
    )
    return attention_type_split_clean_df

def engineer_features(cleaned_only_df, attention_split_df):
    """Takes the dataframe resulting from clean() and seperate_attention_score_variables()
    and constructs variables resembling participants attention standard deviations"""
    # Adding Participants Attention Standard Deviations
    participant_stds = (attention_split_df
                        .groupby("participant_id")[["attention_score_ab", "attention_score_as", "as_minus_ab"]]
                        .std()
                        .reset_index()
                        )
    participant_stds.columns = ["participant_id", "attention_score_ab_std", "attention_score_as_std", "as_minus_ab_std"]
    df_with_stds = pd.merge(left=attention_split_df, right=participant_stds, on="participant_id", how="left")

    # Adding the Total Number of Phases for a Patient
    total_phases_per_participant = (cleaned_only_df
                                    .groupby("participant_id")
                                    .nunique().attention_form_session_phase
                                    .div(2)
                                    )
    df_total_phases = (
        pd.merge(df_with_stds, total_phases_per_participant, "left", on="participant_id")
        .rename(columns={"attention_form_session_phase": "total_phases"})
    )

    # Adding the attention scores and their differences in the first phase of the first recorded session (session 3).
    initial_phase_scores = (df_total_phases
        .loc[lambda df_:
            (df_.session == 3) &
            (df_.phase == 1)
            ]
        [["participant_id", "attention_score_ab", "attention_score_as", "as_minus_ab"]]
        .rename(columns={
            "attention_score_ab": "initial_attention_score_ab",
            "attention_score_as": "initial_attention_score_as",
            "as_minus_ab": "initial_as_minus_ab"})
    )
    df_initial_scores = (pd.merge(
        left=df_total_phases,
        right=initial_phase_scores,
        on="participant_id",
        how="left")
    )

    df_columns_to_convert = [
        "attention_score_ab",
        "attention_score_as",
        "as_minus_ab",
        'total_phases',
        'initial_attention_score_ab',
        'initial_attention_score_as',
        'initial_as_minus_ab',
    ]
    df_initial_scores[df_columns_to_convert] = df_initial_scores[df_columns_to_convert].astype(int)
    return df_initial_scores
