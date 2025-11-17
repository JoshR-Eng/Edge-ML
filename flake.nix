{
	description = "Basic dev environment for edge ML deployment";

	inputs = {
		nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
	};

	outputs = { self, nixpkgs }:
		let
			system = "x86_64-linux"; # Will change on Jetson to "aarch64-linux"
			pkgs = import nixpkgs { inherit system; };
		in {
			# Development Environment
			devShells = {
				"${system}" = {
					default = pkgs.mkShell {

						name = "edge-ml";

						buildInputs = [
						
							# Python 3.11
							pkgs.python311
							pkgs.python311Packages.pip
							pkgs.python311Packages.setuptools
							pkgs.python311Packages.wheel
		
							# Utilities
							pkgs.git
							pkgs.cmake
							pkgs.pkg-config
							pkgs.gcc
						];
	
						shellHook = ''
							echo "Entered Edge-ML Shell"
						'';
					};
				};
			};
		};
}
