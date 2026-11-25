# Installation and updates

After purchase, follow the receipt page's drag-and-drop instructions and accept
adding Superhive as a remote repository. In Blender's repository preferences,
enable **Check for Updates on Startup** and allow online access. New compatible
versions can then be installed through Blender's Extensions interface. This is
Blender's update notification, not a new Rebuilt popup or silent automatic install.
Manual **Install from Disk** remains available, but a local-only installation is
not automatically connected to Superhive. Install from its repository to receive
repository updates; do not enable duplicate copies from different repositories.

For the one-time transition from the old classic Rebuilt:
1. Save your .blend and note any custom Bake memory limit in addon preferences.
2. Disable and remove the classic Rebuilt installation; restart Blender.
3. Install the extension from Superhive, or the matching extension ZIP for testing.
4. Open the saved .blend. Scene settings, including Unity/Unreal/Godot selection,
   remain in the file. Re-enter a custom Bake memory limit if needed: legacy addon
   preferences and extension preferences have different Blender namespaces.

After that transition, repository updates preserve the extension preferences.
Restart Blender after updates because Rebuilt contains native Python libraries.
