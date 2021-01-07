import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


class EntityLocations:
    ENTITY_TYPES = ('eye', 'nail', 'ball')
    ENTITY_ORIENTATIONS = ('stake', 'drill')
    ENTITY_LOCATIONS = ('high', 'low')
    NR_ENTITIES = len(ENTITY_TYPES) * len(ENTITY_ORIENTATIONS) * len(
        ENTITY_LOCATIONS)

    def __init__(self, fname, angle_nr):
        self.fname = fname
        self.angle_nr = angle_nr

        try:
            self._locations = np.load(fname, allow_pickle=True).item()
        except FileNotFoundError:
            self._locations = dict()

    def locations(self):
        return self._locations[self.angle_nr]

    def __getitem__(self, item):
        try:
            value = self._locations[self.angle_nr][item]
        except:
            value = False

        return value

    def __setitem__(self, key, value: tuple):
        print(f"Setting {key} to {value} for projection {self.angle_nr}.")
        if not (self.angle_nr in self._locations):
            self._locations[self.angle_nr] = dict()

        self._locations[self.angle_nr][key] = value
        self.save()

    def save(self):
        np.save(self.fname, self._locations)

    @staticmethod
    def get_iter():
        return itertools.product(EntityLocations.ENTITY_TYPES,
                                 EntityLocations.ENTITY_ORIENTATIONS,
                                 EntityLocations.ENTITY_LOCATIONS)


class Manager:
    def __init__(self, locations, proj):
        self._fig, _ = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        cid = self._fig.canvas.mpl_connect('button_press_event',
                                           self.handle_click)
        self._proj_ax = plt.subplot(1, 2, 1)
        # ax.title(proj_nr)
        self._proj_ax.imshow(proj)

        self._entities = locations
        self._entity_buttons = {}

        self._active_entity = None
        self._active_arrow = None
        self._button_height = 1. / EntityLocations.NR_ENTITIES

        for i, key in enumerate(EntityLocations.get_iter()):
            butt_loc = [0.7, i * self._button_height, .3, self._button_height]
            butt_ax = plt.axes(butt_loc)

            if self._entities[key] is not False:
                coords = self._entities[key]
                coords = f"{coords[0]:.2f}, {coords[1]:.1f}"
            else:
                coords = ''
            butt = Button(butt_ax, f"{key[0]} {key[1]} {key[2]}\n{coords}")

            def click_event_handler(this_button, buttons):
                def _onclick(event):
                    for key, button in buttons.items():
                        if this_button is button['button']:
                            self.set_active(key)
                            this_button.color = 'red'
                        else:
                            button['button'].color = 'gray'

                        plt.gcf().canvas.draw_idle()

                return _onclick

            self._entity_buttons[key] = {'location': butt_loc, 'button': butt}
            butt.on_clicked(click_event_handler(butt, self._entity_buttons))

        self._draw_arrows()
        plt.show()

    def handle_click(self, event):
        if self._active_entity is None:
            print("Click has no effect: no active entity selected.")
            return

        if event.xdata is None or event.ydata is None:
            return  # clicking has no effect outside the image

        if event.inaxes != self._proj_ax:
            return  # or outside of the axes

        if not event.dblclick:
            return  # we need single click to zoom

        self._entities[self._active_entity] = (event.xdata, event.ydata)
        self._draw_arrows()

    def set_active(self, key):
        self._active_entity = key
        self._draw_arrows()

    def _draw_arrows(self):
        ax = plt.subplot(1, 2, 1)
        for key, item in self._entity_buttons.items():
            try:
                item['arrow_annotation'].remove()
            except:
                pass

            if self._entities[key] is False:
                continue

            item['arrow_annotation'] = ax.annotate(
                '', xy=self._entities[key], xycoords='data',
                xytext=[item['location'][0],
                        item['location'][1] + self._button_height / 2],
                textcoords='figure fraction',
                arrowprops=dict(arrowstyle="->", color="red"))

        self._fig.canvas.draw_idle()
        plt.sca(self._proj_ax)
