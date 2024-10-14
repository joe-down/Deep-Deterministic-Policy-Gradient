{
  inputs.nixpkgs.url = "nixpkgs/nixos-24.05";
  outputs = { self, nixpkgs }:
    let
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f:
        nixpkgs.lib.genAttrs supportedSystems
        (system: f { pkgs = import nixpkgs { inherit system; }; });
    in {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            (pkgs.python3.withPackages (python-pkgs: [
              python-pkgs.gymnasium
              python-pkgs.jax
              python-pkgs.jaxlib
              python-pkgs.matplotlib
              python-pkgs.moviepy
              python-pkgs.pybox2d
              python-pkgs.pygame
              python-pkgs.pytestCheckHook
              python-pkgs.scipy
            ]))
            ffmpeg
            opencv4
          ];
        };
      });
    };
}
