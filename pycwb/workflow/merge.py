import glob
import orjson


def merge_catalog(working_dir: str = '.'):
    # config_file = f"{working_dir}/config/user_parameters.yaml"
    # # check if the config file exists
    # if not os.path.exists(config_file):
    #     print(f"Config file {config_file} not found")
    #     return
    #
    # config = Config(config_file)
    # catalog_dir = f"{working_dir}/{config.catalog_dir}"

    # get the list of json files catalog*.json
    catalog_files = glob.glob(f"{working_dir}/catalog/catalog_*.json")
    if len(catalog_files) == 0:
        print("No catalog files found")
        return

    # read the merged file catalog.json
    with open(f"{working_dir}/catalog/catalog.json", 'r') as f:
        merged_catalog = orjson.loads(f.read())

    merged_catalog["events"] = []

    # read the sub catalog files
    for catalog_file in catalog_files:
        with open(catalog_file, 'r') as f:
            catalog = orjson.loads(f.read())
            # append the events
            print(f"Adding {len(catalog['events'])} events from {catalog_file}")
            merged_catalog["events"].extend(catalog["events"])

    print(f"Total number of events: {len(merged_catalog['events'])}")

    # write the merged catalog file
    with open(f"{working_dir}/catalog/catalog.json", 'w') as f:
        f.write(orjson.dumps(merged_catalog, option=orjson.OPT_INDENT_2).decode())