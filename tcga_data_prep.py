
import pandas as pd
import os
import glob
def process_tcga_files(data_dir='./data'):
    expr_files = glob.glob(os.path.join(data_dir, '*_final_expression_filtered.csv'))
    
    for expr_path in expr_files:
        file_prefix = os.path.basename(expr_path).split('_')[0]
        info_path = os.path.join(data_dir, f"{file_prefix}_final_sample_info.csv")
        
        if not os.path.exists(info_path):
            print(f"warning not found: {info_path}")
            continue
            
        print(f"processing: {file_prefix} ...")
        
        df_info = pd.read_csv(info_path)
        mapping = {
            row['sample_id']: f"{row['disease_type']}-{row['stage']}" 
            for _, row in df_info.iterrows()
        }
        
        df_expr = pd.read_csv(expr_path, index_col=0)
        
        labels = [mapping.get(col, "Unknown") for col in df_expr.columns]
        
        df_label_row = pd.DataFrame([labels], columns=df_expr.columns, index=["Group_Info"])
        
        df_ordered = pd.concat([df_label_row, df_expr])
        
        output_filename = expr_path.replace('.csv', '_ordered.csv')
        df_ordered.to_csv(output_filename)
        print(f"saved to: {output_filename}")
        
if __name__ == "__main__":
    process_tcga_files(data_dir='./data')
