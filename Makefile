# nnutil2 - Tensorflow utilities for training neural networks
# Copyright (c) 2019, Abdó Roig-Maranges <abdo.roig@gmail.com>
#
# This file is part of 'nnutil2'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.



# So that we can use newlines in recipes with $(\n)
define \n


endef

# Shell settings
SHELL       := /bin/bash
.SHELLFLAGS := -e -u -c

# Use a single shell
.ONESHELL:

# So we can use $$(variable) on the prerequisites, that expand at matching time.
.SECONDEXPANSION:

# Utility variables
null  :=
space := $(null) #
comma := ,

all:

FORCE:


# -------------------------------------------------------------------------- #
# Files and directories                                                      #
# -------------------------------------------------------------------------- #

NAME         := nnutil2

MAKE_DIR     := make
VENV_DIR     := venv
SRC_DIR      := src
EXT_DIR      := ext
TESTS_DIR    := tests

# Programs
PYTHON       := python
IPYTHON      := ipython
LINT         := pylint -E
VIRTUALENV   := virtualenv
COVERAGE     := coverage

# Deployment paths
PREFIX       := /usr
DOCDIR       := $(PREFIX)/share/doc/$(NAME)
ZSHDIR       := $(PREFIX)/share/zsh/site-functions
BASHDIR      := /etc/bash_completion.d
SHEBANG      := /usr/bin/env $(PYTHON)

PYTHON_LIBS  := $(wildcard $(EXT_DIR)/*)

PYTHON_ENV   := PYTHONPATH=$(SRC_DIR):

# Other variables
ARGS         :=

include $(MAKE_DIR)/utils.mk

SOURCE_FILES := $(shell find $(NAME)/ -name '*.py')



# -------------------------------------------------------------------------- #
# Virtual environment                                                        #
# -------------------------------------------------------------------------- #

$(VENV_DIR)/bin/activate: requirements.txt
	@$(VIRTUALENV) $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r requirements.txt
	if [ -e $@ ]; then touch $@; fi

$(VENV_DIR)/bin/$(NAME): $(VENV_DIR)/bin/activate setup.py $(SOURCE_FILES)
	@$(VENV_DIR)/bin/pip install --editable .

define setup-virtual-environment
	if [ -e "$(HOME)/.bashrc" ]; then
		source "$(HOME)/.bashrc"
	fi

	source "$1/bin/activate"

endef

.PHONY: venv

venv: $(VENV_DIR)/bin/activate $(VENV_DIR)/bin/$(NAME)
	@echo "Entering virtual environment: $(VENV_DIR)"
	$(SHELL) --init-file <(echo "$(call setup-virtual-environment,$(VENV_DIR))")



# -------------------------------------------------------------------------- #
# Targets                                                                    #
# -------------------------------------------------------------------------- #

.PHONY: all build lint

all: build man

build:
	@$(PYTHON) setup.py build --executable="$(SHEBANG)"

lint:
	@$(LINT) $(NAME) *.py


.PHONY: run test coverage

run:
	$(PYTHON) $(NAME) $(ARGS)

test:
	@$(PYTHON) -m unittest -v tests

coverage:
	@$(COVERAGE) run -m unittest tests


.PHONY: clean

clean:
	@$(PYTHON) setup.py clean --all
	find . -name '*.pyc' -exec rm -f {} \;
	find . -name '.cache*' -exec rm -f {} \;
	find . -name '*.html' -exec rm -f {} \;
	rm -Rf $(VENV_DIR) .coverage
	make -C man clean


.PHONY: install

install:
	@$(PYTHON) setup.py install --prefix="$(PREFIX)" --root="$(DESTDIR)"



# -------------------------------------------------------------------------- #
# Source maintenance                                                         #
# -------------------------------------------------------------------------- #

.PHONY: update-template update-copyright

## Update cookiecutter template branch
update-template:
	@python make/cookiecutter-update.py ".cookiecutter.json" template

## Update copyright from file headers
update-copyright:
	@year=$$(date '+%Y')
	git ls-files | while read f; do
		sed -i "1,10{s/Copyright (c) \([0-9]\+\)\(-[0-9]\+\)\?,/Copyright (c) \1-$$year,/}" "$$f"
		sed -i "1,10{s/Copyright (c) $$year-$$year,/Copyright (c) $$year,/}" "$$f"
	done

.PHONY: help

## Print Makefile documentation
help:
	@perl -0 -nle 'printf("%-25s - %s\n", "$$2", "$$1") while m/^##\s*([^\r\n]+)\n^([\w-]+):[^=]/gm' \
		$(MAKEFILE_LIST) | sort
	printf "\n"
	perl -0 -nle 'printf("%-25s - %s\n", "$$2=", "$$1") while m/^##\s*([^\r\n]+)\n^([\w-]+)\s*:=/gm' \
		$(MAKEFILE_LIST) | sort
