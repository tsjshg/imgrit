from dataclasses import dataclass
from random import sample
from math import isclose

from PIL import Image, ImageDraw
import numpy as np
from scipy.spatial import KDTree, Voronoi

# kmeans is clearly a bottle neck of this library
# todo: replace faster implementation such as by Rust
try:
    from sklearn.cluster import KMeans

    # I have scikit-learn
    HAVE_SKL = True
except ImportError:
    from scipy.cluster.vq import kmeans2

    HAVE_SKL = False

ZERO_TOL = 1.0e-12


@dataclass
class Pixel:
    """A pixel with coordinate and the distance to the Voronoi site"""

    x: int
    y: int
    distance: float

    def to_array(self):
        return np.array([self.x, self.y], np.int64)


def solve(vec, p, v, on="x"):
    """法線ベクトルvecと通る点pを受け取って、onで指定された値をもとめる
    onがxならvはyの値で、算出されたxを返すということ。

    Check vec[0] and vec[1] is not zero before using this.
    """
    if on == "x":
        return p[0] - vec[1] * (v - p[1]) / vec[0]
    return p[1] - vec[0] * (v - p[0]) / vec[1]


def find_edge_point(ridge_point_pair, regions, vor, x_max=600, y_max=400):
    """ボロノイ領域の境界線を描くための関数(描画用)

    2つのボロノイ領域を分ける境界線（ボロノイ境界？）を表現する線分のうち、
    片方のインデックスが-1のものを受け取り、画像の境界線とどこで交わるかを探してその点の座標を返す関数
    """
    # −1ではない方のボロノイ点の座標 (たぶん、計算しなくてもよさそう -> 先頭が常に−1？）
    i = ridge_point_pair[1] if ridge_point_pair[0] == -1 else ridge_point_pair[0]
    # voronoi vertex
    # 座標を入れ換えるために配列を逆順にする
    vv = vor.vertices[i][::-1]
    # この点が画像の内部かどうかのチェック
    # 画像の外のボロノイ点も含まれているため
    if not (0 <= vv[0] <= x_max and 0 <= vv[1] <= y_max):
        return None

    # 領域のインデックスから母点の座標を取得
    # 座標系を入れ換える
    a = vor.points[regions[0]][::-1]
    b = vor.points[regions[1]][::-1]
    # 垂直2等分線の方向を表すベクトル
    v1 = b - a
    # explore the point across Voronoi edge and the image boundary
    temp = []
    if isclose(v1[0], 0.0, abs_tol=ZERO_TOL):  # parallel along with x-axis
        temp = [(0, vv[1]), (x_max, vv[1])]
    elif isclose(v1[1], 0.0, abs_tol=ZERO_TOL):  # parallel along with ｙ-axis
        temp = [(vv[0], 0), (vv[1], y_max)]
    else:  # pass the zero div check
        for v, on in [(0, "x"), (0, "y"), (x_max, "y"), (y_max, "x")]:
            ret_val = solve(v1, vv, v, on)
            if on == "x" and 0 <= ret_val <= x_max:
                # 画像の範囲に入っているｘ座標
                temp.append((ret_val, v))
            elif 0 <= ret_val <= y_max:
                # 画像の範囲に入っているｙ座標
                temp.append((v, ret_val))
    # 制御点に近い方を返す（これでいいのか？）
    if np.linalg.norm(temp[0] - a) < np.linalg.norm(temp[1] - a):
        return tuple([temp[0], tuple(vv)])
    return tuple([temp[1], tuple(vv)])


class VoronoiRegion:
    """A Voronoi region"""

    def __init__(self, site, color):
        """
        site: 2D array-like
            coordinate of the site
        color:
        """
        self.site = site
        # the color of this region
        self.color = color
        # Pixels that belong to this region
        self.followers = []

    def append(self, pixel):
        if not isinstance(pixel, Pixel):
            raise ValueError("pixel must be Pixel object.")
        self.followers.append(pixel)

    def size(self):
        # number of pixels in this region
        return len(self.followers) + 1

    def __sub__(self, obj):
        """color difference"""
        return np.abs(
            np.array(self.color, dtype=np.int32) - np.array(obj.color, dtype=np.int32)
        ).sum()


class VoronoiImage:
    """元画像とボロノイ制御点を保持してボロノイ画像としての最低限の機能を有する
    制御点の最適化などはこのクラスのインスタンスを受け取って、制御点の位置を最適化する設計が良いかと
    """

    def __init__(self, img, sites):
        """
        img: PIL image file
        sites: int or array-like. specify the number of Voronoi sites or concrete one
        """
        self.img = img
        # to NumPy array
        self.img_array = np.array(self.img)
        # todo: take care of alpha value
        # now ignore them
        if self.img_array.shape[2] == 4:
            self.img_array = self.img_array[:, :, :3]
        # axis0: height of the image
        # axis1: width of the image
        self.axis0, self.axis1, self.color_channels = self.img_array.shape
        # coordinate of all the pixel
        self.grid_array = [
            (x, y) for x in np.arange(self.axis0) for y in np.arange(self.axis1)
        ]
        if isinstance(sites, int):
            # sites are given by number
            self.sites_num = sites
            self.sites = self.init_sites()
        elif isinstance(sites, (list, tuple, np.array)):
            # sites are given by specific values
            self.sites_num = len(sites)
            # to tuple
            temp = [tuple(v) for v in sites]
            if not all([self._is_in_frame(v) for v in temp]):
                raise ValueError("all sites must be within the image.")
            self.sites = tuple(temp)
        else:
            raise TypeError("sites must be int or 2D array-list object")
        # the followings are set after calling create_Voronoi_image method
        # Voronoi mosaic art
        self.voronoi_img = None
        # difference between the input pixel values and their Voronoi site pixel values
        self.error = None
        # each voronoi region
        self.regions = None

    def get_error(self):
        if self.error is None:
            return np.inf
        return self.error.sum()

    def init_sites(self):
        """determine Voronoi sites randomly"""
        return sample(self.grid_array, self.sites_num)

    def _is_in_frame(self, p):
        """within the frame?"""
        return 0 <= p[0] < self.axis0 and 0 <= p[1] < self.axis1

    def _convert_boundaries(self, boundaries, width_dict):
        """todo: understand and test this 色の差を線の太さに変換する関数"""
        data = np.array([v[1] for v in boundaries]).reshape(-1, 1)
        k = len(width_dict)
        if HAVE_SKL:
            kmeans = KMeans(n_clusters=k).fit(data)
            pred = kmeans.labels_
            centroid = kmeans.cluster_centers_
        else:
            centroid, pred = kmeans2(data, k, minit="++")
        # クラスターの中心の並びを調べる
        class_to_width = {v: i for i, v in enumerate(np.argsort(centroid.flatten()))}
        for i in range(len(boundaries)):
            # boundaries[i][1] = class_to_width[pred[i]] + 1
            boundaries[i][1] = width_dict[class_to_width[pred[i]]]
        return boundaries

    def create_Voronoi_image(self, boundary=True, with_sites=False, line_width=1):
        """Voronoi image

        boundary: boolean
            with boundary lines or not
        with_sites: boolean
            if true put 'x' on control point
        line_width: int or dict
            if dict is given like {0:1, 1:5, 2:10} line width are changed based on the color differences between neighbor Voronoit regions

        Returns: PIL image
        """
        kd_tree = KDTree(self.sites)
        # make Resions from self.sites
        self.regions = [
            VoronoiRegion(s, self.img_array[s[0], s[1]]) for s in self.sites
        ]
        # if line_width is dict the line width will be changed
        change_width = isinstance(line_width, dict)
        # make Voronoi diagram to create the mosaic art
        res = kd_tree.query(self.grid_array)
        for i, p in enumerate(self.grid_array):
            if res[0][i] == 0:
                # skip site point
                continue
            pixel = Pixel(p[0], p[1], res[0][i])
            # res[1][i] is the index of sites
            self.regions[res[1][i]].append(pixel)
        voronoi_art = self.img_array.copy()
        # initialize the error
        # error = np.zeros(3, np.float64)
        # don't change the color of the site point
        for region in self.regions:
            for p in region.followers:
                voronoi_art[p.x, p.y] = self.img_array[region.site[0], region.site[1]]
                # error += np.abs(
                #     np.array(voronoi_art[p.x, p.y], dtype=np.int32)
                #     - np.array(self.img_array[p.x, p.y], dtype=np.int32)
                # )
        # self.error = np.sum(error) / (self.axis0 * self.axis1)
        # if don't need the boundaries this is the goal.
        img = Image.fromarray(voronoi_art)
        if not boundary:
            self.voronoi_img = img
            return img
        # make Voronoi diagram with SciPy function
        vor = Voronoi(self.sites)
        draw = ImageDraw.Draw(img)
        # to change the width of boundaries depend on the color difference
        boundaries = []  # [[[start, end], color diff], ...]
        for k, v in vor.ridge_dict.items():
            if -1 in v:
                idx = find_edge_point(v, k, vor, self.axis1, self.axis0)
                if idx is None:
                    continue
            else:
                idx = tuple([(a[1], a[0]) for a in vor.vertices[v]])
                # 画像の外にある点が指定されても、画角に収まる線だけが描画されることが判明
                # 画角の外側のボロノイ点と結ばれて線が描かれることもある。
            if change_width:
                # 中点を計算（座標を逆にするのが正しいみたい）
                m = ((idx[0][1] + idx[1][1]) * 0.5, (idx[0][0] + idx[1][0]) * 0.5)
                # これら2つを分ける境界線
                res = kd_tree.query(m, k=2)
                boundaries.append(
                    [idx, self.regions[res[1][0]] - self.regions[res[1][1]]]
                )
            else:
                boundaries.append([idx, line_width])
        if change_width:
            boundaries = self._convert_boundaries(boundaries, line_width)
        for idx, width in boundaries:
            try:
                draw.line(idx, fill=(0, 0, 0), width=width)
            except:
                # What's happened?
                print(idx)
        if with_sites:
            for vv in vor.points:
                draw.text((vv[1], vv[0]), "x")
        self.voronoi_img = img
        return img


class KMeansImage:
    def __init__(self, img_file):
        if isinstance(img_file, str):
            self.img = Image.open(img_file)
        else:
            self.img = img_file
        self.img_array = np.array(self.img)
        self.h, self.w, self.channel = self.img_array.shape
        # X, Y
        data = np.array([[x, y] for x in np.arange(self.h) for y in np.arange(self.w)])
        # R G B L
        data = np.concatenate(
            [
                data,
                np.array([self.img_array[:, :, 0].flatten()]).T,  # R
                np.array([self.img_array[:, :, 1].flatten()]).T,  # G
                np.array([self.img_array[:, :, 2].flatten()]).T,  # B
                np.array([np.array(self.img.convert("L")).flatten()]).T,
            ],
            axis=1,
        )
        self.img_minmax = data / data.max(
            axis=0
        )  # all values are positive so this make the input minmax-scaled data
        self.voronoi_img_instance = None

    def voronoi_img(
        self,
        num_sites,
        with_sites=False,
        line_width=1,
        boundary=True,
        mode="color",
    ):
        if mode == "color":
            # X, Y, R, G, B
            temp = self.img_minmax[:, [0, 1, 2, 3, 4]]
        else:  # black and white
            temp = self.img_minmax[:, [0, 1, 5]]
        if HAVE_SKL:
            kmeans = KMeans(n_clusters=num_sites).fit(temp)
            centroid = kmeans.cluster_centers_
        else:
            centroid, _ = kmeans2(temp, num_sites, minit="++")
        # depends on the order of self.img_df column names
        sites = [
            tuple(v) for v in (centroid[:, [0, 1]] * [self.h, self.w]).astype(np.int32)
        ]
        self.voronoi_img_instance = VoronoiImage(self.img, sites)
        voronoi_img = self.voronoi_img_instance.create_Voronoi_image(
            with_sites=with_sites, line_width=line_width, boundary=boundary
        )
        return voronoi_img

    def clustered_img(self, num_sites):
        # X, Y, R, G, B
        input_data = self.img_minmax[:, [0, 1, 2, 3, 4]]
        if HAVE_SKL:
            kmeans = KMeans(n_clusters=num_sites).fit(input_data)
            labels = kmeans.labels_
            centroid = kmeans.cluster_centers_
        else:
            centroid, labels = kmeans2(input_data, num_sites, minit="++")
        cluster_centers = [
            v for v in (centroid[:, 2:] * [255, 255, 255]).astype(np.int32)
        ]
        temp = []
        for c in range(3):
            temp.append(
                np.array([cluster_centers[i][c] for i in labels])
                .astype(np.uint8)
                .reshape(self.h, self.w)
            )
        res_img_array = np.dstack(temp)
        return Image.fromarray(res_img_array)


def voronoi_mosaic(img, num_regions=20, line_width=1, mode="L"):
    """create Voronoi mosaic art from your input image.

    img: str (file path) or PIL.Image instance
    num_regions: int (default=20)
        number of retions. larger value may take few minutes to convet the images.
    line_width: int (default=2)
        line width for Voronoi boundaries
    mode: 'L' or 'color'
        default is 'L'
    """
    img = KMeansImage(img)
    voronoi_img = img.voronoi_img(num_regions, line_width=line_width, mode=mode)
    return voronoi_img


def warhol_effect(img, n_clusters=5):
    """create Warhol effect art from your input image.

    img: str (file path) or PIL.Image instance
    n_clusters: int (default=5)
        number of colors to use
    """
    img = KMeansImage(img)
    return img.clustered_img(n_clusters)
