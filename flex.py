# Ignore deprecation warnings
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Import required modules
import pandas
import flexmatcher

# Load our three data sources
source1 = pandas.read_csv("06-08-2020.csv")
source2 = pandas.read_csv("10-21-2020.csv")
source3 = pandas.read_csv("co-est2019-alldata.csv", encoding = 'latin-1')

# Examine the shape of each source
print(f"Shape of first source: {source1.shape[0]} rows,"
      f" {source1.shape[1]} columns")
print(f"Shape of second source: {source2.shape[0]} rows,"
      f" {source2.shape[1]} columns")
print(f"Shape of third source: {source3.shape[0]} rows,"
      f" {source3.shape[1]} columns")
print()

# Check for duplicate entries in each source
num_dupe1 = len(source1[source1.duplicated(["Combined_Key"])])
num_dupe2 = len(source2[source2.duplicated(["Combined_Key"])])
num_dupe3 = len(source3[source3.duplicated(["COUNTY", "STNAME"])])
print(f"Number of duplicates in first source: {num_dupe1}")
print(f"Number of duplicates in second source: {num_dupe2}")
print(f"Number of duplicates in third source: {num_dupe3}")
print()

# Filter columns and remove NaNs for our two training sets
train1 = source1[["Admin2", "Province_State"]].dropna()
train2 = source2[["Admin2", "Province_State"]].dropna()

# Create an attribute name map
mapping = {"Admin2": "county",
           "Province_State": "state"}

# Create a FlexMatcher object and train it
schema_list = [train1, train2]
mapping_list = [mapping, mapping]
fm = flexmatcher.FlexMatcher(schema_list, mapping_list,
                             sample_size = 100)
fm.train()

# Remove numeric data from test set
test = source3.select_dtypes(exclude = "number")

# Make a prediction and print it
pred = fm.make_prediction(test)
print(f"Result: {pred}")
