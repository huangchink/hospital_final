# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import os

from psychopy import visual, core, event

from gazefollower import GazeFollower

if __name__ == '__main__':
    win = visual.Window(fullscr=True, color='white')
    gaze_cursor = visual.ShapeStim(win, vertices='circle', size=(100, 100), lineWidth=5,
                                   fillColor=None, lineColor=(0, 255, 0), colorSpace='rgb255',
                                   units='pix')
    # init GazeFollower
    gf = GazeFollower()
    gf.preview(win=win)
    gf.calibrate(win=win)
    gf.start_sampling()
    core.wait(0.1)

    # images need to show
    img_folder = 'images'
    images = ['grid.jpg']

    # show the images one by one in a loop, press a ENTER key to exit the program
    for _img in images:
        # show the image on screen
        im = visual.ImageStim(win, os.path.join(img_folder, _img))
        im.draw()
        win.flip()
        # send a trigger to record in the eye movement data to mark picture onset
        gf.send_trigger(202)

        # now lets show the gaze point, press any key to close the window
        got_key = False
        max_duration = 20.0
        t_start = core.getTime()
        event.clearEvents()  # clear all cached events if there were any
        gx, gy = -65536, -65536
        while not (got_key or (core.getTime() - t_start) >= max_duration):

            # check keyboard events
            if event.getKeys():
                got_key = True

            # redraw the screen
            win.color = (0, 0, 0)
            im.draw()

            # show gaze cursor, when formal experiment
            # you can remove this code
            # ++++++++++++++++++++++++
            gaze_info = gf.get_gaze_info()

            screen_width = win.size[0]
            screen_height = win.size[1]
            if gaze_info and gaze_info.status:
                raw_gx = int(gaze_info.filtered_gaze_coordinates[0])
                raw_gy = int(gaze_info.filtered_gaze_coordinates[1])
                if (raw_gx != -65536) and (raw_gy != -65536):
                    gx = raw_gx - screen_width // 2
                    gy = (screen_height // 2) - raw_gy
                    gaze_cursor.pos = (gx, gy)
                    gaze_cursor.draw()
                else:
                    gaze_cursor.pos = (9999, 9999)
            else:
                gaze_cursor.pos = (9999, 9999)
            # ++++++++++++++++++++++++

            # flip the frame
            win.flip()

    core.wait(0.1)
    gf.stop_sampling()
    win.close()
    # save the sample data to file
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    # save data
    file_name = "free_viewing_psychopy_demo.csv"
    gf.save_data(os.path.join(data_dir, file_name))
    gf.release()
    # quit
    core.quit()