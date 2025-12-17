import pandas as pd
import nannyml as nml

REFERENCE = "splits/validate.csv"
CURRENT = "splits/test.csv"
CHUNK_SIZE = 500


def main():
    print("\n=== NANNYML DATA DRIFT ANALYSIS (TERMINAL REPORT) ===\n")

    ref = pd.read_csv(REFERENCE)
    cur = pd.read_csv(CURRENT)

    calc = nml.UnivariateDriftCalculator(
        column_names=ref.columns.tolist(),
        chunk_size=CHUNK_SIZE,
    )

    calc.fit(ref)
    result = calc.calculate(cur)

    df = result.to_df()

    drift_events = []

    # Iterate over features
    for feature in df.columns.levels[0]:
        if feature in ["index", "chunk"]:
            continue

        alert_col = (feature, "jensen_shannon", "alert")
        value_col = (feature, "jensen_shannon", "value")

        if alert_col not in df.columns:
            continue

        drifted_chunks = df[df[alert_col]]

        for idx, row in drifted_chunks.iterrows():
            drift_events.append({
                "feature": feature,
                "chunk": row[("chunk", "chunk", "chunk_index")],
                "value": row[value_col],
                "threshold": row[(feature, "jensen_shannon", "upper_threshold")],
            })

    if not drift_events:
        print("NO DATA DRIFT DETECTED ACROSS ALL FEATURES\n")
        return

    print(f"DATA DRIFT DETECTED IN {len(set(d['feature'] for d in drift_events))} FEATURES\n")

    for event in drift_events:
        print(
            f"- Feature: {event['feature']}, "
            f"Chunk: {int(event['chunk'])}, "
            f"JS Distance: {event['value']:.4f}, "
            f"Threshold: {event['threshold']:.4f}"
        )

    print("\n=== END OF REPORT ===\n")


if __name__ == "__main__":
    main()
