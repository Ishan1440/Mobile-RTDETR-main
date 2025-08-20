{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-23-11.url = "github:Nixos/nixpkgs/nixos-23.11";
    systems = {
      url = "github:nix-systems/default";
    };
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    nixpkgs-23-11,
    ...
  } @ inputs:
    inputs.flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {
        config,
        system,
        lib,
        ...
      }: let
        pkgs = import nixpkgs {inherit system;};
        pkgs-old = import nixpkgs-23-11 {inherit system;};
      in {
        devShells.default = pkgs.mkShell {
          venvDir = ".venv";
          strictDeps = false;
          packages = with pkgs-old.python310Packages; [
            torch-bin
            torchvision-bin
            pycocotools
            cython_3
            matplotlib
            scipy
            onnx
            pycocotools
            pyyaml
            torch-tb-profiler
            tensorboard
            fvcore
          ];
        };
      };
    };
}
