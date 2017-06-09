import time
from copy import deepcopy
from mapping.extraction import Extraction
from road_extraction.road_extractor import extract_road, post_process_extraction
from utils.distance import euclidean_distance


class MappingTool:
    def __init__(self, fig, ax, im, img, img_ref=None, mapping_style=None, rt_proposals=None, close=None):
        # statistics
        self.extension_count = 0
        self.correction_count = 0
        self.removal_count = 0
        self.start_time = time.time()
        self.stop_time = None
        self.nb_extractions = 0
        self.extraction_time = 0

        # model
        self.extraction = None
        self.extractions = []

        # functional attributes
        self.history = []
        self.click_position = None
        self.picked_point = None
        self.picked_line = None
        # the last road proposal to avoid recomputation
        self.proposal = None
        self.position = None
        self.prev_position = None
        # function to call when finished
        self.close = close

        # state
        self.correcting = False
        self.ref_displayed = False
        # allows to set an unchangeable mapping style
        self.mapping_style = mapping_style
        if mapping_style is not None:
            if mapping_style == 'manual':
                self.manual = True
            elif mapping_style == 'semi-automatic':
                self.manual = False
            else:
                raise ValueError("Mapping style not supported (should be 'manual', 'semi-automatic' or None)")
        else:
            self.manual = False
        # show real-time road or line proposals
        if rt_proposals is not None:
            self.rt_proposals = rt_proposals
            self.lock_rt_toggle = True
        else:
            self.rt_proposals = False
            self.lock_rt_toggle = False

        self.stop = False

        # view
        self.fig = fig
        self.ax = ax
        self.im = im
        self.img = img
        self.img_ref = img_ref
        self.artists = None
        self.all_artists = []
        # scatters last point of current extraction
        self.lp_scatter = None
        # plots road proposals
        self.rp_plot = None
        # background for smooth corrections
        self.corr_background = None
        # background for smooth road proposals
        self.prop_background = None

        self.set_title()
        self.new_extraction()
        self.connect()

        # timer to wait for the mouse pointer to be idle before calculating a proposal
        timer = self.fig.canvas.new_timer(interval=200)
        timer.add_callback(self.on_timeout)
        timer.start()

    def get_title(self):
        if self.rt_proposals:
            title = 'Real-time'
            if self.manual:
                title += ' manual mapping'
            else:
                title += ' semi-automatic mapping'
        else:
            if self.manual:
                title = 'Manual mapping'
            else:
                title = 'Semi-automatic mapping'
        if self.correcting:
            title += ' (correcting)'
        else:
            title += ' (extending)'
        return title

    def set_title(self, title=None):
        if title is None:
            title = self.get_title()
        self.fig.canvas.set_window_title(title)

    def set_extending_state(self):
        self.correcting = False
        self.click_position = None
        self.picked_point = None
        self.picked_line = None
        self.corr_background = None
        self.update_artist_last_point()
        self.set_title()

    def set_correcting_state(self):
        self.correcting = True
        self.click_position = None
        self.picked_point = None
        self.picked_line = None
        self.corr_background = None
        self.update_artist_last_point()
        self.set_title()

    def get_all_correct_colors(self, exclude_points=None):
        acc = []
        for e in self.extractions:
            for cp in e.correct_points:
                include = True
                for p in exclude_points:
                    if cp == p:
                        include = False
                        break
                if include:
                    acc.append(self.img[cp[1]][cp[0]])
        return acc

    def extract_road(self, start, end, proposal=False):
        if proposal:
            start_time = time.time()
            extraction = extract_road(self.img, start, end, self.get_all_correct_colors([start, end]))
            self.extraction_time += time.time() - start_time
            self.nb_extractions += 1
            return extraction
        else:
            if self.rt_proposals and self.proposal is not None:
                smoothed_ext, points, extraction, cost_map = self.proposal
                ds = euclidean_distance(start, extraction[0])
                de = euclidean_distance(end, extraction[-1])
                if ds < 4.5 and de < 4.5:
                    # _, points = post_process_extraction(extraction, cost_map)
                    points[0] = start
                    points[-1] = end
                    return smoothed_ext, points, extraction, cost_map
        start_time = time.time()
        extraction = extract_road(self.img, start, end, self.get_all_correct_colors([start, end]))
        self.extraction_time += time.time() - start_time
        self.nb_extractions += 1
        return extraction

    def new_extraction(self):
        for i, extraction in enumerate(self.extractions):
            if len(extraction.points) == 0:
                del self.extractions[i]
                del self.all_artists[i]

        self.extraction = Extraction()
        self.extractions.append(self.extraction)

        l_plot, = self.ax.plot([], [], c='yellow', linewidth=2, picker=2)
        p_scatter = self.ax.scatter([], [], c='yellow', s=20, picker=2)
        cp_scatter = self.ax.scatter([], [], c='yellow', edgecolors='orange', s=40, linewidths=1.5, picker=2)

        self.artists = {'lines': l_plot,
                        'points': p_scatter,
                        'correct_points': cp_scatter}
        self.all_artists.append(self.artists)

        # must be recreated to make sure it's on top
        if self.lp_scatter is not None:
            self.lp_scatter.remove()
        self.lp_scatter = self.ax.scatter([], [], c='yellow', edgecolors='red', s=50, linewidths=1.5, picker=2)
        if self.rp_plot is not None:
            self.rp_plot.remove()
        self.rp_plot, = self.ax.plot([], [], c='yellow', linewidth=2)
        self.rp_plot.set_animated(True)
        self.prop_background = None

        self.fig.canvas.draw()
        self.clear_history()

    def set_extraction(self, i):
        self.extraction = self.extractions[i]
        self.artists = self.all_artists[i]
        self.update_artists()

    def extend_extraction(self, point):
        self.save_model()
        start = self.extraction.get_last()
        if start:
            if self.manual:
                points = [start, point]
            else:
                title = self.get_title()
                title += ' - computing shortest path'
                self.set_title(title)
                _, points, _, _ = self.extract_road(start, point)
                self.set_title()
            self.extraction.extend(points, point)
        else:
            self.extraction.extend([point], point)
        self.extension_count += 1
        self.update_artists()
        # if start and not self.manual and not self.rt_proposals:
        #    self.new_extraction()

    def correct_extraction(self, new_point):
        self.save_model()
        pair = self.extraction.get_closest_correct_point_pair(new_point)
        if pair:
            start, end = pair
            if self.manual:
                points = [start, new_point, end]
            else:
                title = self.get_title()
                title += ' - recomputing shortest paths'
                self.set_title(title)
                _, points1, _, _ = self.extract_road(start, new_point)
                _, points2, _, _ = self.extract_road(new_point, end)
                points = points1[:-1] + points2
                self.set_title()
            self.extraction.correct_segment(points, start, new_point, end)
            self.correction_count += 1
            self.update_artists()
        else:
            self.update_artist_last_point()

    def correct_point(self, point, new_point):
        self.save_model()
        if point != new_point:
            indices = []
            for i, extraction in enumerate(self.extractions):
                corrected = extraction.correct_point(point, new_point)
                if corrected:
                    indices.append(i)
            if len(indices) > 0:
                self.correction_count += 1
            self.update_all_artists(indices)

    def insert_point(self, point, index, index_extraction):
        self.save_model()
        if index_extraction == -1:
            self.extraction.insert_point(point, index)
            self.update_artists()
        else:
            extraction = self.extractions[index_extraction]
            artists = self.all_artists[index_extraction]
            extraction.insert_point(point, index)
            self.update_artists(extraction, artists)
        self.correction_count += 1

    def remove_point(self, point):
        self.save_model()
        indices = []
        for i, extraction in enumerate(self.extractions):
            is_correct = point in extraction.correct_points
            removed = extraction.remove_point(point)
            if removed:
                indices.append(i)
                # if is_correct:
                #     if len(extraction.correct_points) > 1:
                #         points_new = []
                #         for i in range(len(extraction.correct_points) - 1):
                #             start = extraction.correct_points[i]
                #             end = extraction.correct_points[i + 1]
                #             if self.manual:
                #                 points = [start, end]
                #             else:
                #                 title = self.get_title()
                #                 title += ' - recomputing shortest paths'
                #                 self.set_title(title)
                #                 start_time = time.time()
                #                 _, points, _, _ = self.extract_road(start, end)
                #                 self.extraction_time += time.time() - start_time
                #                 self.nb_extractions += 1
                #                 self.set_title()
                #             points_new = points_new[:-1] + points
                #         extraction.points = points_new
        if len(indices) > 0:
            self.removal_count += 1
        self.update_all_artists(indices)

    def plot_road_proposal(self, end_rp):
        start_rp = self.extraction.get_last()
        if start_rp and self.rt_proposals and self.prop_background is not None:
            if self.manual:
                sx, sy = start_rp
                ex, ey = end_rp
                self.rp_plot.set_data([sx, ex], [sy, ey])
            else:
                title = self.get_title()
                title += ' - computing shortest path'
                self.set_title(title)
                smoothed_ext, points, extraction, cost_map = self.extract_road(start_rp, end_rp, proposal=True)
                px = [x for (x, y) in points]
                py = [y for (x, y) in points]
                self.rp_plot.set_data(px, py)
                self.proposal = smoothed_ext, points, extraction, cost_map
                self.set_title()
            self.fig.canvas.restore_region(self.prop_background)
            self.ax.draw_artist(self.rp_plot)
            self.fig.canvas.blit(self.ax.bbox)

    def plot_point(self, point, blit=False, restore_background=False):
        self.lp_scatter.set_offsets([point])
        if blit:
            if restore_background and self.corr_background is not None:
                self.fig.canvas.restore_region(self.corr_background)
            self.ax.draw_artist(self.lp_scatter)
            self.fig.canvas.blit(self.ax.bbox)
        else:
            self.fig.canvas.draw()

    def update_artist_last_point(self, draw=True):
        if self.correcting:
            self.lp_scatter.set_offsets(self.extraction.correct_points)
        else:
            last_point = self.extraction.get_last()
            if last_point:
                self.lp_scatter.set_offsets([last_point])
            else:
                self.lp_scatter.set_offsets([])
        if draw:
            self.fig.canvas.draw()

    def update_artists(self, extraction=None, artists=None, draw=True):
        if extraction is None or artists is None:
            extraction = self.extraction
            artists = self.artists

        l_plot = artists['lines']
        p_scatter = artists['points']
        cp_scatter = artists['correct_points']

        px = [x for (x, y) in extraction.points]
        py = [y for (x, y) in extraction.points]

        l_plot.set_data(px, py)
        p_scatter.set_offsets(extraction.points)
        cp_scatter.set_offsets(extraction.correct_points)

        self.update_artist_last_point(draw=draw)

        if self.rt_proposals:
            self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def update_all_artists(self, indices=[]):
        if len(indices) > 0:
            for i in indices:
                extraction = self.extractions[i]
                artists = self.all_artists[i]
                self.update_artists(extraction, artists, draw=False)
            self.fig.canvas.draw()
        else:
            for i, extraction in enumerate(self.extractions):
                artists = self.all_artists[i]
                self.update_artists(extraction, artists, draw=False)
            self.fig.canvas.draw()

        if self.rt_proposals:
            self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def save_model(self):
        i = self.extractions.index(self.extraction)
        extractions = deepcopy(self.extractions)
        extraction = extractions[i]
        self.history.append((extractions, extraction))

    def undo_last_edit(self):
        if len(self.history) > 0:
            self.extractions, self.extraction = self.history.pop()
            self.update_all_artists()

    def clear_history(self):
        self.history = []

    def get_number_of_points(self):
        np = 0
        ncp = 0
        for extraction in self.extractions:
            np += len(extraction.points)
            ncp += len(extraction.correct_points)
        return np, ncp

    def get_statistics(self):
        np, ncp = self.get_number_of_points()
        if self.stop_time is None:
            total_time = time.time() - self.start_time
        else:
            total_time = self.stop_time - self.start_time

        return {'Segment count': len(self.extractions),
                'Point count': np,
                'Given point count': ncp,
                'Extension count': self.extension_count,
                'Correction count': self.correction_count,
                'Removal count': self.removal_count,
                'Click count': self.extension_count + self.correction_count + self.removal_count,
                'Extraction count': self.nb_extractions,
                'Extraction time': round(self.extraction_time, 2),
                'Total time spend': round(total_time, 2)}

    def print_statistics(self):
        np, ncp = self.get_number_of_points()
        if self.stop_time is None:
            total_time = time.time() - self.start_time
        else:
            total_time = self.stop_time - self.start_time

        print('------------')
        print('Number of segments:', len(self.extractions))
        print('Number of points:', np)
        print('Number of given points:', ncp)
        print('Number of extensions:', self.extension_count)
        print('Number of corrections:', self.correction_count)
        print('Number of removals:', self.removal_count)
        print('Number of successful clicks:', self.extension_count + self.correction_count + self.removal_count)
        print('Number of extractions:', self.nb_extractions)
        print('Total extraction time: %s seconds' % round(self.extraction_time, 2))
        print('Total time spend: %s seconds' % round(total_time, 2))
        print('------------')

    def on_click(self, event):
        if event.xdata is None or event.ydata is None: return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        if not ((0 <= x < self.img.shape[1]) and (0 <= y < self.img.shape[0])): return
        point = x, y

        e = -1
        picked_point = None
        contains, info = self.artists['points'].contains(event)
        if contains:
            ind = info['ind']
            if len(ind) > 0:
                picked_point = self.extraction.points[ind[0]]
        else:
            for e, artists in enumerate(self.all_artists):
                p_scatter = artists['points']
                contains, info = p_scatter.contains(event)
                if contains:
                    ind = info['ind']
                    if len(ind) > 0:
                        picked_point = self.extractions[e].points[ind[0]]
                        break

        index_line = None
        if picked_point is None:
            e = -1
            contains, info = self.artists['lines'].contains(event)
            if contains:
                ind = info['ind']
                if len(ind) > 0:
                    index_line = ind[0]
            else:
                for e, artists in enumerate(self.all_artists):
                    l_scatter = artists['lines']
                    contains, info = l_scatter.contains(event)
                    if contains:
                        ind = info['ind']
                        if len(ind) > 0:
                            index_line = ind[0]
                            break

        if event.button == 1:
            if event.key == 'control':
                if (picked_point is not None or index_line is not None) and e >= 0:
                    self.set_extraction(e)
            elif self.correcting:
                if picked_point is not None:
                    self.click_position = (event.xdata, event.ydata)
                    self.picked_point = picked_point
                    self.corr_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                    self.plot_point(picked_point, blit=True)
                elif index_line is not None:
                    self.click_position = (event.xdata, event.ydata)
                    self.picked_line = (e, index_line)
                    self.corr_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                    self.plot_point(self.click_position, blit=True)
                else:
                    self.plot_point(point, blit=True)
                    self.correct_extraction(point)
            else:
                if picked_point is not None:
                    point = picked_point
                self.plot_point(point, blit=True)
                self.extend_extraction(point)
        elif event.button == 3:
            if picked_point is not None:
                if e < 0:
                    self.remove_point(picked_point)
                else:
                    # self.set_extraction(e)
                    self.remove_point(picked_point)

    def on_timeout(self):
        if self.correcting or not self.rt_proposals or self.position is None: return
        start = self.extraction.get_last()
        if not start: return

        # do not compute proposal if we already have a proposal ending in this point with a margin < 2 pixels
        if self.proposal is not None:
            _, points, _, _ = self.proposal
            m = euclidean_distance(self.position, points[-1])
            if m < 2:
                return

        # if distance between start and end is small: compute proposal
        # else: wait until mouse pointer is stable for 1 timeout period
        d = euclidean_distance(start, self.position)
        if d < 200 or self.prev_position is None:
            self.plot_road_proposal(self.position)
        else:
            m = euclidean_distance(self.position, self.prev_position)
            if m < 2:
                self.plot_road_proposal(self.position)

        self.prev_position = self.position

    def on_motion(self, event):
        if event.xdata is None or event.ydata is None: return
        if self.correcting:
            if self.click_position is None: return
            if self.picked_point is None and self.picked_line is None: return

            x, y = event.xdata, event.ydata
            cx, cy = self.click_position
            dx, dy = x - cx, y - cy

            if self.picked_point is not None:
                px, py = self.picked_point
                nx, ny = px + dx, py + dy
            else:
                nx, ny = cx + dx, cy + dy

            if (0 <= nx < self.img.shape[1]) and (0 <= ny < self.img.shape[0]):
                self.plot_point((nx, ny), blit=True, restore_background=True)
        elif self.rt_proposals and self.extraction.get_last():
            x, y = event.xdata, event.ydata
            x, y = int(round(x)), int(round(y))
            if self.manual:
                self.plot_road_proposal((x, y))
            else:
                self.position = x, y

    def on_release(self, event):
        if not self.correcting: return
        if self.click_position is None: return
        if self.picked_point is None and self.picked_line is None: return
        if event.xdata is None or event.ydata is None: return

        x, y = event.xdata, event.ydata
        cx, cy = self.click_position

        if x == cx and y == cy:
            self.update_artist_last_point()
        else:
            if self.picked_point is not None:
                px, py = self.picked_point
                dx, dy = x - cx, y - cy
                nx, ny = int(round(px + dx)), int(round(py + dy))

                if (0 <= nx < self.img.shape[1]) and (0 <= ny < self.img.shape[0]):
                    if self.picked_point != (nx, ny):
                        self.correct_point(self.picked_point, (nx, ny))
                    else:
                        self.update_artist_last_point()
            else:
                dx, dy = x - cx, y - cy
                nx, ny = int(round(cx + dx)), int(round(cy + dy))

                if (0 <= nx < self.img.shape[1]) and (0 <= ny < self.img.shape[0]):
                    e, i = self.picked_line
                    self.insert_point((nx, ny), i + 1, e)

        self.click_position = None
        self.picked_point = None
        self.picked_line = None
        self.corr_background = None

    def on_key_press(self, event):
        if event.key == ' ':
            self.set_extending_state()
            self.new_extraction()
        elif event.key == 'c' or event.key == 'C':
            if self.correcting:
                self.set_extending_state()
            else:
                self.set_correcting_state()
        elif event.key == 'u' or event.key == 'U':
            self.undo_last_edit()
        elif event.key == 'r' or event.key == 'R':
            if not self.lock_rt_toggle:
                if self.rt_proposals:
                    self.rt_proposals = False
                    self.position = None
                    self.prev_position = None
                    self.prop_background = None
                    self.rp_plot.set_data([], [])
                    self.fig.canvas.draw()
                else:
                    self.rt_proposals = True
                    self.position = None
                    self.prev_position = None
                    self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                self.set_title()
        elif event.key == 'h' or event.key == 'H':
            if self.img_ref is not None:
                if self.ref_displayed:
                    self.im.set_data(self.img)
                    self.ref_displayed = False
                else:
                    self.im.set_data(self.img_ref)
                    self.ref_displayed = True
                self.fig.canvas.draw()
                if self.prop_background is not None:
                    self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        elif event.key == 'm' or event.key == 'M':
            if self.mapping_style is None:
                if self.manual:
                    self.manual = False
                else:
                    self.manual = True
                self.set_title()
        elif event.key == 'i' or event.key == 'I':
            self.print_statistics()
        elif event.key == 'enter':
            if self.close is not None:
                self.stop_time = time.time()
                self.disconnect()
                self.close()
        elif event.key == 'escape':
            # self.stop = True
            # if self.close is not None:
            #     self.disconnect()
            #     self.close()
            self.set_extending_state()
            self.new_extraction()
        elif event.key == 'delete':
            self.extraction.reset()
            self.update_artists()

        if event.key == 'up':
            ylim = self.ax.get_ylim()
            c = 0.1 * (ylim[0] - ylim[1])
            ylim_new = ylim[0] - c, ylim[1] - c
            self.ax.set_ylim(ylim_new)
            self.fig.canvas.draw()
            if self.rt_proposals:
                self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        elif event.key == 'down':
            ylim = self.ax.get_ylim()
            c = 0.1 * (ylim[0] - ylim[1])
            ylim_new = ylim[0] + c, ylim[1] + c
            self.ax.set_ylim(ylim_new)
            self.fig.canvas.draw()
            if self.rt_proposals:
                self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        elif event.key == 'left':
            xlim = self.ax.get_xlim()
            c = 0.1 * (xlim[1] - xlim[0])
            xlim_new = xlim[0] - c, xlim[1] - c
            self.ax.set_xlim(xlim_new)
            self.fig.canvas.draw()
            if self.rt_proposals:
                self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        elif event.key == 'right':
            self.prop_background = None
            xlim = self.ax.get_xlim()
            c = 0.1 * (xlim[1] - xlim[0])
            xlim_new = xlim[0] + c, xlim[1] + c
            self.ax.set_xlim(xlim_new)
            self.fig.canvas.draw()
            if self.rt_proposals:
                self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        elif event.key == '0':
            self.ax.set_xlim(-0.5, self.img.shape[1] - 0.5)
            self.ax.set_ylim(self.img.shape[0] - 0.5, -0.5)
            self.fig.canvas.draw()
            if self.rt_proposals:
                self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    # Source: stackoverflow.com/a/19829987
    def zoom(self, event):
        if event.xdata is None or event.ydata is None: return
        s = 2

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if event.button == 'up':
            scale_factor = 1 / s
        elif event.button == 'down':
            if cur_xlim[1] - cur_xlim[0] > 0.9 * self.img.shape[0]:
                self.ax.set_xlim((-0.5, self.img.shape[1] - 0.5))
                self.ax.set_ylim((self.img.shape[0] - 0.5, -0.5))
                self.fig.canvas.draw()
                if self.rt_proposals:
                    self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
                return
            else:
                scale_factor = s
        else:
            scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.fig.canvas.draw()

        if self.rt_proposals:
            self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def on_resize(self, event):
        if self.rt_proposals:
            self.fig.canvas.draw()
            self.prop_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def connect(self):
        self.cid_on_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_on_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_on_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_on_key_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_on_scroll = self.fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.cid_on_resize = self.fig.canvas.mpl_connect('resize_event', self.on_resize)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid_on_click)
        self.fig.canvas.mpl_disconnect(self.cid_on_motion)
        self.fig.canvas.mpl_disconnect(self.cid_on_release)
        self.fig.canvas.mpl_disconnect(self.cid_on_key_press)
        self.fig.canvas.mpl_disconnect(self.cid_on_scroll)
        self.fig.canvas.mpl_disconnect(self.cid_on_resize)
