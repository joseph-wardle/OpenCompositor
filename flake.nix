{
  # The development environment on NixOS: `direnv allow` once, or `nix develop`. Cargo is still
  # the whole interface; this only supplies what Cargo cannot.
  description = "OpenCompositor and OpenNG development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { nixpkgs, ... }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      qt = pkgs.qt6.qtbase;
      # wgpu and winit load these at run time; the Vulkan drivers are the system's.
      loaded = pkgs.lib.makeLibraryPath (with pkgs; [
        vulkan-loader
        wayland
        libxkbcommon
        libx11
        libxcursor
        libxi
      ]);
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          # Installs what rust-toolchain.toml names, here and in compositor-kernels/shaders.
          rustup
          trunk
          spirv-tools # spirv-dis, for comparing a rebuilt kernels.spv as CI does
        ];

        # The shell links Qt, private headers included. The linker records where it lives, so the
        # binary runs against the Qt it was built with.
        buildInputs = [ qt ];
        QMAKE = "${qt}/bin/qmake";

        # This Qt's plugins alone: the desktop session names its own, built for another Qt.
        QT_PLUGIN_PATH = "${qt}/lib/qt-6/plugins";

        shellHook = ''
          export LD_LIBRARY_PATH="${loaded}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };
    };
}
