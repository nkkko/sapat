# Build and Publish Process

Follow these steps to build and publish a new version of the package:

1. Update Version Number
   - Open the file `sapat/__version__.py`
   - Increment the version number according to semantic versioning (e.g., from '0.1.0' to '0.1.1')

2. Update CHANGELOG.md
   - Add a new entry for the new version
   - List all notable changes, additions, and fixes

3. Commit Changes
   - Stage the changed files:
     ```
     git add sapat/__version__.py CHANGELOG.md
     ```
   - Commit the changes:
     ```
     git commit -m "Bump version to X.Y.Z and update CHANGELOG"
     ```

4. Create a Git Tag
   - Create a new tag with the version number:
     ```
     git tag vX.Y.Z
     ```
   - Push the tag to the remote repository:
     ```
     git push origin vX.Y.Z
     ```

5. Build the Package
   - Ensure you have the latest build tools:
     ```
     pip install --upgrade setuptools wheel twine
     ```
   - Build the distribution packages:
     ```
     python setup.py sdist bdist_wheel
     ```

6. Check the Distribution
   - Use twine to check the distribution:
     ```
     twine check dist/*
     ```

7. Upload to TestPyPI (Optional)
   - Upload the distribution to TestPyPI:
     ```
     twine upload --repository-url https://test.pypi.org/legacy/ dist/*
     ```
   - Test the installation from TestPyPI:
     ```
     pip install --index-url https://test.pypi.org/simple/ --no-deps sapat
     ```

8. Upload to PyPI
   - Upload the distribution to PyPI:
     ```
     twine upload dist/*
     ```

9. Verify Installation
   - In a new virtual environment, install the package from PyPI:
     ```
     pip install sapat
     ```
   - Run a quick test to ensure the package is working correctly

10. Update Documentation
    - Update any version-specific documentation on your project's website or README

11. Create a GitHub Release
    - Go to your GitHub repository
    - Create a new release, using the tag you created earlier
    - Include the CHANGELOG entries for this version in the release notes

12. Announce the Release
    - Inform your users about the new release through your preferred channels (e.g., mailing list, social media, blog post)

Remember to replace 'X.Y.Z' with your actual version number throughout this process.