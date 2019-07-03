# Copyright (c) 2019, Abdó Roig-Maranges <abdo.roig@gmail.com>
# All rights reserved.
#
# This file is part of 'nnutil2'.
#
# This file may be modified and distributed under the terms of the 3-clause BSD
# license. See the LICENSE file for details.


##
# first <list>
#
# Returns first word in list
##
first = $(firstword $1)

##
# tail <list>
#
# Returns tail of the list excluding first word
##
tail = $(wordlist 2,$(words $1),$1)

##
# map <func> <list>
#
# Apply a function to all the elements in a list
##
map = $(strip $(foreach a,$2,$(call $1,$a)))

##
# append <element> <list>
#
# Append an element at the end of a list
##
append = $(strip $(foreach a,$2,$(call $1,$a)))

##
# assoc <key> <list1> <list2>
#
# Match key in list1 and return the element of list2 at the same position.
##
assoc = $(strip $(if $2,$(if ifeq($(call first $2),$1),$(call first $3),$(assoc $1,(call rest $2),(call rest $3)))))

##
# splitdir
#
# Split directory into a list of its components
##
splitdir      = $(subst /, ,$1)

##
# make-relative <base> <path>
#
# Make <path> relative respect to <base>. Note, paths must exist in the filesystem.
##
make-relative = $(shell realpath -L --relative-to $1 $2)

##
# write-if-chaned-cmd <str> <path>
#
# Shell command to write <str> to <path> only if contents of file are different from <str>.
##
define write-if-changed-cmd
	if [ ! -f "$2" ] || [ ! "$1" = "`cat $2 2> /dev/null`" ]; then
		echo "$1" > "$2";
	fi
endef

# A comma. We need this since we cannot replace a comma with subst, because it
# is the argument separator.
, := ,

##
# make-pattern-list <list of patterns>
#
# Convert a comma-separated list of *-wildcard patterns to GNU make patterns.
##
make-pattern-list = $(subst $(,), ,$(subst *,%,$1))

##
# rwildcard <directory> <*-pattern>
#
# Recursive wildcard
##
rwildcard         = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

##
# value-if-defined <variable name> <default value>
#
# Get the value of a variable if defined, or a default value otherwise.
##
value-if-defined  = $(if $(strip $(filter undefined,$(origin $1))),$2,$($1))

