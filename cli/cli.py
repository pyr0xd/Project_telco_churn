import argparse
import sys
import os
import glob
import json

# python -m cli.cli

def create_parser():
    #Create all parsers
    parser = argparse.ArgumentParser(description="CLI tool for model training interface", prog="")
    subparsers = parser.add_subparsers(dest="command", help="choose command")

    train_parser = subparsers.add_parser("train-base", help="Run train/val with GridSearch and MLflow-experiment")
    train_parser.add_argument(
        "--models", 
        nargs="+", 
        default=["LogisticRegression", "RandomForest"], 
        help="Models that will be trained"
    )
    
    runs_parser = subparsers.add_parser("runs", help="Previous models and params MLflow")
    runs_parser.add_argument(
        "--limit", 
        type=int, 
        default=5, 
        help="Previous runs (standard: 5)"
    )
    
    eval_parser = subparsers.add_parser("train-prod", help="Train model on full train+val and evaluate on test set")
    eval_parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["LogisticRegression", "RandomForest"],
        help="Which model to train for production and evaluate"
    )

    subparsers.add_parser("fetch", help="Fetch dataset from database")

    return parser

def main():
    parser = create_parser()
    print("================================================================")
    print(" Welcome to the CLI model training interface!")
    print(" Commandlist:")
    print(" fetch - fetch the data from datase")
    print(" train-base - train a model")
    print(" train-prod - train a model")
    print(" runs - previous model training runs")
    print(" ")
    print(" help - get command list")
    print(" exit")
    print("================================================================")

    while True:
        try:
            line = input("churn-cli> ").strip()
            if not line:
                continue
            
            parts = line.split()
            
            if parts[0] == "exit":
                print("Exiting CLI")
                break
                
            if parts[0] == "help":
                parser.print_help()
                continue

            args = parser.parse_args(parts)

            if args.command == "train-base":
                print(f"Starting training for models: {args.models}...")
                from training.train_test import run_training
                run_training(selected_models=args.models)
                print("Training complete! Models saved and logged to MLflow.\n")
                
            elif args.command == "fetch":
                print("Fetching data from the database")
                from training.dataset_load import fetch_dataset
                df = fetch_dataset()
                print(f"Completted! Fetched {len(df)} rows.\n")
            
            elif args.command == "runs":
                print("fetching MLflow history...")
                try:
                    import mlflow
                    from mlflow.tracking import MlflowClient
                    
                    runs = mlflow.search_runs(experiment_names=["basemodel_test"])
                    
                    if runs.empty:
                        print("No previous runs.\n")
                    else:
                        print(f"\nLatest models (Max {args.limit} st):")
                        print("-" * 70)
                        
                        cols_to_show = ['run_id', 'params.model_type', 'metrics.val_f1_score', 'metrics.val_accuracy']
                        available_cols = [c for c in cols_to_show if c in runs.columns]
                        
                        print(runs[available_cols].head(args.limit).to_string(index=False))
                        print("-" * 70 + "\n")
                        
                except Exception as e:
                    print(f"Could not find MLflow history: {e}\n")
            
            elif args.command == "train-prod":
                baseline_dirs = sorted(glob.glob("artifacts/baseline_*"), reverse=True)
                if not baseline_dirs:
                    print("No immutable baseline directories (artifacts/baseline_*) found. Run 'train-base' first.\n")
                    continue
                
                latest_baseline = baseline_dirs[0]
                print(f"Using baseline artifact directory: {latest_baseline}")
                
                model_files = glob.glob(os.path.join(latest_baseline, "*_best.joblib"))
                if not model_files:
                    print("No model files (*_best.joblib) found in baseline directory.\n")
                    continue
                
                payload_path = os.path.join(latest_baseline, "telco_metrics_payload.json")
                payload_data = {}
                if os.path.exists(payload_path):
                    try:
                        with open(payload_path, "r", encoding="utf-8") as f:
                            payload_data = json.load(f)
                    except Exception:
                        pass

                available_models = []
                print("\nAvailable baseline models:")
                print("-" * 40)
                for i, path in enumerate(model_files, 1):
                    filename = os.path.basename(path)
                    mkey = filename.replace("_best.joblib", "")
                    disp_name = "LogisticRegression" if "logistic" in mkey else ("RandomForest" if "random" in mkey else mkey)
                    available_models.append((i, disp_name, mkey, path))
                    print(f"[{i}] {disp_name} (File: {filename})")
                print("-" * 40)
                
                choice = input("Select the number of the model you would like to train for production: ").strip()
                
                try:
                    selected_idx = int(choice) - 1
                    disp_name, mkey, path = available_models[selected_idx], available_models[selected_idx], available_models[selected_idx]
                    
                    print(f"Model: {disp_name} selected. Starting production training...")
                    
                    from training.train_production import run_production_training
                    
                    # Pull best params from baseline payload if present
                    default_params = payload_data.get(mkey, {}).get("best_params", {})
                    if not default_params:
                        if disp_name == "LogisticRegression":
                            default_params = {'C': 10.0}
                        elif disp_name == "RandomForest":
                            default_params = {'n_estimators': 50, 'max_depth': 10}
                            
                    run_production_training(model_name=disp_name, custom_params=default_params)
                    
                except (ValueError, IndexError):
                    print("Error: Invalid selection.\n")
            
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()