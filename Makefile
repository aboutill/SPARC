# =============================================================================
# Makefile: part of sparc package.
#
# Installs the SPARC CLI and pulls the sparc Docker image.
#
# Usage:
#   make install                       # install to ~/.local/bin, pull image
#   make install PREFIX=/usr/local/bin # system-wide install
#   make pull-image                    # (re-)pull the Docker image only
#   make uninstall
# =============================================================================

.PHONY: install uninstall pull-image

PREFIX ?= $(HOME)/.local/bin
DOCKER_IMG ?= aboutill/sparc:v1.0.0

install: pull-image
	mkdir -p $(PREFIX)
	ln -sf $(CURDIR)/scripts/SPARC.bash $(PREFIX)/SPARC
	@case ":$$PATH:" in \
	  *":$(PREFIX):"*) ;; \
	  *) $(MAKE) -s path-hint ;; \
	esac
	
path-hint:
	@case "$$SHELL" in \
	  */zsh) rc="$(HOME)/.zshrc" ;; \
	  *) rc="$(HOME)/.bashrc" ;; \
	esac; \
	echo ""; \
	echo "Note: $(PREFIX) is not on your PATH."; \
	echo "  Add this line to $$rc (or your shell's equivalent):"; \
	echo "    export PATH=\"$(PREFIX):\$$PATH\""; \
	echo "  then run: source $$rc"; \
	echo ""

pull-image:
	docker pull $(DOCKER_IMG) || echo "Warning: could not pull $(DOCKER_IMG) -- it will be pulled automatically on first run instead."

uninstall:
	rm -f $(PREFIX)/SPARC
