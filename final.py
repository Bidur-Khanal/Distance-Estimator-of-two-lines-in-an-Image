#!/usr/bin/env python

import pygtk
pygtk.require('2.0')
import gtk
import take_reference as ref
import camera_take_pic as cm
import line


class HelloWorld2:

    # Our new improved callback.  The data passed to this method
    # is printed to stdout.
    def callback(self, widget, data):
        if data==1:
            cm.take_image("ref5.png")
            #print "%d was pressed" % data
        if data==2:
            cm.take_image("line5.png")
            #print "%d was pressed" % data
        if data==3:
            ref.find_ref("ref5.png")
            #print "%d was pressed" % data

        if data==4:
            line.find_bend("line5.png")
            #print "%d was pressed" % data
        

    # another callback
    def delete_event(self, widget, event, data=None):
        gtk.main_quit()
        return False

    def __init__(self):
        # Create a new window
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)

        # This is a new call, which just sets the title of our
        # new window to "Hello Buttons!"
        self.window.set_title("Test Program")

        # Here we just set a handler for delete_event that immediately
        # exits GTK.
        self.window.connect("delete_event", self.delete_event)

        # Sets the border width of the window.
        self.window.set_border_width(10)

        # We create a box to pack widgets into.  This is described in detail
        # in the "packing" section. The box is not really visible, it
        # is just used as a tool to arrange widgets.
        self.box1 = gtk.HBox(False, 0)

        # Put the box into the main window.
        self.window.add(self.box1)

        # Creates a new button with the label "Button 1".
        self.button1 = gtk.Button("Capture reference image")

        # Now when the button is clicked, we call the "callback" method
        # with a pointer to "button 1" as its argument
        self.button1.connect("clicked", self.callback, 1)

        # Instead of add(), we pack this button into the invisible
        # box, which has been packed into the window.
        self.box1.pack_start(self.button1, True, True, 0)

        # Always remember this step, this tells GTK that our preparation for
        # this button is complete, and it can now be displayed.
        self.button1.show()

        # Do these same steps again to create a second button
        self.button2 = gtk.Button("Capture parallel line image")

        # Call the same callback method with a different argument,
        # passing a pointer to "button 2" instead.
        self.button2.connect("clicked", self.callback, 2)

        self.box1.pack_start(self.button2, True, True, 0)

        # The order in which we show the buttons is not really important, but I
        # recommend showing the window last, so it all pops up at once.
        self.button2.show()

        # Do these same steps again to create a third button
        self.button3 = gtk.Button("Find Dimension to Pixel ratio")

        # Call the same callback method with a different argument,
        # passing a pointer to "button 3" instead.
        self.button3.connect("clicked", self.callback, 3)

        self.box1.pack_start(self.button3, True, True, 0)

        # The order in which we show the buttons is not really important, but I
        # recommend showing the window last, so it all pops up at once.
        self.button3.show()

        # Do these same steps again to create a fourth button
        self.button4 = gtk.Button("Find distance")

        # Call the same callback method with a different argument,
        # passing a pointer to "button 4" instead.
        self.button4.connect("clicked", self.callback, 4)

        self.box1.pack_start(self.button4, True, True, 0)

        # The order in which we show the buttons is not really important, but I
        # recommend showing the window last, so it all pops up at once.
        self.button4.show()
        self.box1.show()
        self.window.show()

def main():
    gtk.main()

if __name__ == "__main__":
    hello = HelloWorld2()
    main()
