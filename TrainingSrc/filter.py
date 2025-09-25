import pandas as pd
import sys
import os

def filter_csv_columns(input_file_path, output_file_path=None):
    """
    Read a CSV file and create a new one with only 'cuerpo' and 'titular' columns.
    
    Args:
        input_file_path (str): Path to the input CSV file
        output_file_path (str): Path for the output CSV file (optional)
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file_path):
            print(f"Error: File '{input_file_path}' not found.")
            return False
        
        # Read the CSV file
        print(f"Reading CSV file: {input_file_path}")
        df = pd.read_csv(input_file_path)
        
        print(f"Original file has {len(df)} rows and {len(df.columns)} columns")
        print(f"Available columns: {list(df.columns)}")
        
        # Check if required columns exist
        required_columns = ['cuerpo', 'titular']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print("Please check the column names in your CSV file.")
            return False
        
        # Filter to keep only the required columns
        df_filtered = df[required_columns].copy()
        
        # Generate output file path if not provided
        if output_file_path is None:
            # Create new filename by adding '_filtered' before the extension
            base_name = os.path.splitext(input_file_path)[0]
            extension = os.path.splitext(input_file_path)[1]
            output_file_path = f"{base_name}_filtered{extension}"
        
        # Save the filtered data
        print(f"Creating filtered CSV with columns: {required_columns}")
        df_filtered.to_csv(output_file_path, index=False)
        
        print(f"Success! New file created: {output_file_path}")
        print(f"New file has {len(df_filtered)} rows and {len(df_filtered.columns)} columns")
        
        # Show a preview of the data
        print("\nPreview of filtered data:")
        print(df_filtered.head())
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False

def main():
    print("CSV Column Filter - Extract 'cuerpo' and 'titular' columns")
    print("=" * 55)
    
    # Get input file path from command line argument or user input
    if len(sys.argv) > 1:
        input_file_path = sys.argv[1]
    else:
        input_file_path = "IAExample.csv"
    
    # Remove quotes if user wrapped path in quotes
    input_file_path = input_file_path.strip('"\'')
    
    # Get output file path (optional)
    output_file_path = None
    if len(sys.argv) > 2:
        output_file_path = sys.argv[2]
    else:
        output_input = "IAFinal.csv"
        if output_input:
            output_file_path = output_input.strip('"\'')
    
    # Process the file
    success = filter_csv_columns(input_file_path, output_file_path)
    
    if success:
        print("\nOperation completed successfully!")
    else:
        print("\nOperation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()