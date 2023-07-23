PYTHON = python3

PACKAGE_NAME = hedges
BUILD_DIR = build
DIST_DIR = dist

publish: $(BUILD_DIR)
	$(PYTHON) -m twine upload $(DIST_DIR)/*

$(BUILD_DIR): clean
	$(PYTHON) -m build

clean:
	rm -rf $(DIST_DIR) $(BUILD_DIR) $(PACKAGE_NAME).egg-info

