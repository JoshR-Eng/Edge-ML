{
	description = "Edge ML dev environment for Nvidia Jetson";

	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    utils.url  = "github:numtide/flake-utils";
  };

	outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
		  let
			  system = "aarch64-linux";
			  pkgs = import nixpkgs { inherit system; };

		  in {
			  # Development Environment
			  devShells.default = pkgs.mkShell {
            name = "edge-ml";

            buildInputs = with pkgs; [
          
              # Python 3.11
              python311
              python311Packages.pip
              python311Packages.virtualenv
              python311Packages.setuptools
              python311Packages.wheel
              python311Packages.numpy
  
              # Utilities
              git
              cmake
              gcc
              htop
              wget

              # Dependencies
              zlib
              libxml2
              libxslt
              pkg-config
            ];

            shellHook = ''
              echo "--- Edge-ML Shell Activated ---"
              echo "    System: ${system}"

              # Load into Python venv for pip packages 
              if [ ! -d ".venv" ]; then
                echo "Creating virtual environment..."
                python -m venv .venv
              fi
              source .venv/bin/activate
              
              # Instructions if testing on PC or on Edge
              if [[ "${system}" == "aarch64-linux" ]] then
                echo ""
                echo "Device: NVIDIA JETSON"
                echo "installing hardware dependencies via pip"
                pip install jetson-stats
                pip install onnxruntime-gpu --extra-index-url https://pypi.ngc.nvidia.com
              
              else 
                echo ""
                echo "Device: HOST"
                echo "Installing dependencies for testing"
                pip install onnxruntime 
              fi 
            '';
				  };
			  };
      );		
    };
}
