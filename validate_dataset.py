import argparse, pandas as pd
from data.schema import inspect_schema, runtime_columns, validate_no_removed_columns

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/raw/intrusex_bh.csv")
    a=ap.parse_args()
    df=pd.read_csv(a.data)
    validate_no_removed_columns(df)
    s=inspect_schema(df)
    print("rows:",len(df))
    print("runs:",df["run_id"].nunique())
    print("runtime feature count:",s.dimension)
    print("runtime features:",s.runtime_columns)
    print("stage counts:",df["stage"].value_counts().sort_index().to_dict())
    print("validation: OK")
if __name__=="__main__": main()
