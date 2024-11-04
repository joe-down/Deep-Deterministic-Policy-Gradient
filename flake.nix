{
  inputs.nixpkgs.url = "nixpkgs/nixos-24.05";
  outputs = { self, nixpkgs }:
    let
      supportedSystems =
        [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f:
        nixpkgs.lib.genAttrs supportedSystems (system:
          f {
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          });
    in {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            (pkgs.python3.withPackages (python-pkgs: [
              #Gymnasium
              python-pkgs.gymnasium
              python-pkgs.pybox2d
              python-pkgs.pygame
              #Pytorch
              python-pkgs.torch-bin
              #matplotlib
              python-pkgs.matplotlib
              #tqdm
              python-pkgs.tqdm
            ]))
            ffmpeg
            opencv4
          ];
        };
      });
    };
}
