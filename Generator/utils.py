import yaml

# import avro.schema as schema
# from avro.datafile import DataFileReader, DataFileWriter
# from avro.io import DatumReader, DatumWriter


def load_config_file():
    """ Function for loading parameters for running the LLM
    return loaded dictionary with data from the configuration file"""
    with open("launching_config.yml", "r") as file:
        config_data = yaml.safe_load(file)
    return config_data


def load_file_with_prompts(file_name: str):
    """ Function for loading the prompts dataset for processing.

    :param return: list with loaded prompts
    """
    with open(file_name, "r") as file:
        prompts = [line.rstrip() for line in file]
    return prompts


def save_prompts(file_name: str, prompts_list: list, mode: str = "a"):
    """ Function for saving the prompts stored in the prompts list

    :param file_name: a string with the name of the file that will be loaded
    :param prompts_list: a list with strings (generated prompts)
    :param mode: mode for writing the file: 'a', 'w'
    """
    with open(file_name, mode) as file:
        for p in prompts_list:
            file.write("%s" % p)


# def load_prompts_avro(file_name: str):
#     avro_reader = DataFileReader(open(file_name, "rb"), DatumReader())
#     data = [d for d in avro_reader]
#     avro_reader.close()
#     return data
#
#
# def save_prompts_avro(prompts: list, file_name: str, schema_file: str = "../avro_schema.avsc"):
#     avro_schema = schema.parse(open(schema_file, "rb").read())
#     avro_writer = DataFileWriter(open(file_name, "wb"), DatumWriter(), avro_schema)
#     for p in prompts:
#         avro_writer.append({"name": p})
#     avro_writer.close()

