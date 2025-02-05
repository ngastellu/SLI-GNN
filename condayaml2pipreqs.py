import yaml

with open("environment.yml") as file_handle:
    # Use safe_load instead of load
    environment_data = yaml.safe_load(file_handle)

with open("requirements.txt", "w") as file_handle:
    for dependency in environment_data["dependencies"]:
        # Ensure the dependencies are correctly formatted (handle versions as well)
        if isinstance(dependency, str):  # If it's a basic dependency with a version
            package_name, package_version = dependency.split("=")
            file_handle.write("{} == {}\n".format(package_name, package_version))
        elif isinstance(dependency, dict):  # If it's a package with extra info (like channels)
            # Example: handle dependencies with specific conditions like channels
            for sub_dependency in dependency.get('pip', []):
                print(sub_dependency)
                print(sub_dependency.split('='))
                package_name, package_version = sub_dependency.split("=")
                file_handle.write("{} == {}\n".format(package_name, package_version))
