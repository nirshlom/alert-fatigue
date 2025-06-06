import pandas as pd

def main():
    # Read the ignore other orders file
    ignore_orders_path = 'alert_analysis/data/ignore_other_orders.csv'
    ignore_orders_df = pd.read_csv(ignore_orders_path).rename(columns={'Order_ID_new_update': 'drug_order_id'})

    # Read the main data file
    main_data_path = 'alert_analysis/data/main_data_2022/df_main_active_adult_renamed.csv'
    main_df = pd.read_csv(main_data_path)

    # Keep only required columns from main data
    main_df = main_df[['drug_order_id', 'response_reasons_other_text']]

    # Merge the dataframes by order_id
    merged_df = pd.merge(
        main_df,
        ignore_orders_df,
        on='drug_order_id',
        how='inner'
    )

    # Save the merged result
    output_path = 'alert_analysis/data/main_data_2022/ignore_orders_with_text.csv'
    merged_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✅ Merged data saved to {output_path}")
    print(f"Shape of merged data: {merged_df.shape}")
    
    return merged_df

if __name__ == "__main__":
    main()
