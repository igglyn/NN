{ pkgs, lib, config, inputs, ... }:

{
  packages = with pkgs.python313Packages; [
    numpy
  ];

  languages.python = {
    enable = true;
    package = pkgs.python313;
    #venv.enable = true;
    #uv.enable = true;
  };
}
