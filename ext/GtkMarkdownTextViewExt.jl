module GtkMarkdownTextViewExt

#   This file is part of Cartan.jl
#   It is licensed under the GPL license
#   Cartan Copyright (C) 2026 Michael Reed
#       _           _                         _
#      | |         | |                       | |
#   ___| |__   __ _| | ___ __ __ ___   ____ _| | __ _
#  / __| '_ \ / _` | |/ / '__/ _` \ \ / / _` | |/ _` |
# | (__| | | | (_| |   <| | | (_| |\ V / (_| | | (_| |
#  \___|_| |_|\__,_|_|\_\_|  \__,_| \_/ \__,_|_|\__,_|
#
#   https://github.com/chakravala
#   https://crucialflow.com

using Cartan
isdefined(Cartan, :Requires) ? (import Cartan: GtkMarkdownTextView) : (using GtkMarkdownTextView)
import GtkMarkdownTextView: Gtk4

function Cartan.gtkplotmd(md::String,gtk)
    pushfirst!(gtk[][1].child.child,GtkMarkdownTextView.MarkdownTextView(md))
    return gtk
end

end # module
