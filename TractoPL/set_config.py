from TractoPL.configuration import load_hcp_bundle_mapping, load_tool_config


def set_config():
    """Load external-tool configuration without changing the process environment."""
    return load_tool_config()

def get_HCP_bundle_names(bundle_name=None,inverse=False):
    """
    Get the list of bundle names from the HCP dataset.
    """
    mapping = load_hcp_bundle_mapping()
    if bundle_name is None:
        if not inverse:
            return mapping
        else:
            return {value: key for key, value in mapping.items()}
    else:
        if not inverse:
            return mapping.get(bundle_name,bundle_name)
        else:
            inv_mapping = {v: k for k, v in mapping.items()}
            return inv_mapping.get(bundle_name,bundle_name)

if __name__ == '__main__':
    set_config()
    print(get_HCP_bundle_names())
    print(get_HCP_bundle_names("OR_right",inverse=True))
    print(get_HCP_bundle_names("ORright",inverse=False))
    