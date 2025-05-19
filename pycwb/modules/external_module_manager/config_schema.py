schema = {
    "type": "object",
    "properties": {
        "target_dir": {
            "type": ["string", "null"],
            "description": "Directory to check for the module. Defaults to None."
        },
        "modules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the module to check."
                    },
                    "repo_url": {
                        "type": ["string", "null"],
                        "description": "URL of the repository to pull from."
                    },
                    "version": {
                        "type": ["string", "null"],
                        "description": "Version of the module to check."
                    },
                    "module_path": {
                        "type": ["string", "null"],
                        "description": "Path to the module."
                    },
                },
                "required": ["name"],
            },
            "description": "List of modules to check or pull."
        }
    },
}