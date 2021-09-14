from typing import List, Tuple
import numpy as np
from numpy import float32, ndarray
from habitat_sim import registry as registry
from habitat_sim.utils.data import PoseExtractor, ImageExtractor
from scipy.ndimage import binary_erosion
import imageio
import quaternion as qt


# Pose extractor code
@registry.register_pose_extractor(name="sphere_sampler")
class CylinderPoseExtractor(PoseExtractor):
    def __init__(
        self,
        topdown_views: List[Tuple[str, str, Tuple[float32, float32, float32]]],
        meters_per_pixel: float = 0.1,
    ) -> None:
        super().__init__(topdown_views, meters_per_pixel)

    def extract_poses(
        self, view: ndarray, fp: str
    ) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]]:
        # Determine the physical spacing between each camera position
        height, width = view.shape
        # We can modify this to be user-defined later
        dist = min(height, width) // 10
        cam_height = 3
        # Create a grid of camera positions
        n_gridpoints_width, n_gridpoints_height = (
            width // dist - 1,
            height // dist - 1,
        )
        floorplan = binary_erosion(view, iterations=3)
        # groups of xz points sampled from accessible areas in the scene
        gridpoints = []
        # Scene reachability mask with bounds away from walls.
        for h in range(n_gridpoints_height):
            for w in range(n_gridpoints_width):
                point = (dist + h * dist, dist + w * dist)
                if self._valid_point(*point, floorplan):
                    gridpoints.append(point)
        # Generate a pose for vertical slices of the cylindrical panorama
        poses = []
        for row, col in gridpoints:
            position = (col, cam_height, row)
            points_of_interest = self._panorama_extraction(position, view, dist)
            poses.extend([(eye, lap, fp) for eye, lap in points_of_interest])
        # Returns poses in 3D cartesian coordinate system
        return poses

    def _convert_to_scene_coordinate_system(
        self,
        poses: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], str]],
        ref_point: Tuple[float32, float32, float32],
    ) -> List[Tuple[Tuple[int, int], qt.quaternion, str]]:
        # Convert from topdown map coordinate system to that of the scene
        start_point = np.array(ref_point)
        converted_poses = []
        for i, pose in enumerate(poses):
            pos, look_at_point, filepath = pose

            new_pos = start_point + np.array(pos) * self.meters_per_pixel
            new_lap = start_point + np.array(look_at_point) * self.meters_per_pixel
            displacement = new_lap - new_pos

            rot = qt.from_rotation_matrix(
                lookAt(np.array([0, 0, 0]), displacement, np.array([0, 1, 0]))[:3, :3]
            )
            converted_poses.append((new_pos, rot, filepath))

        return converted_poses

    def _panorama_extraction(
        self, point: Tuple[int, int, int], view: ndarray, dist: int
    ) -> List[Tuple[Tuple[float, float,float], Tuple[float,float,float]]]:
        phi_range = np.linspace(0,180,200)
        theta_range = np.linspace(0,360, endpoint=False)

        sphere_coords = np.array(np.meshgrid(phi_range, theta_range)).T.reshape(-1,2)
        samples = []
        radius = 2

        # one pose for each pixel column in the panoramic image
        point = np.array(point)
        for coord in sphere_coords:
            offset = (np.array(sphere_to_cartesian(coord[1],coord[0])) * radius)
            eye =  point + offset
            lap = eye + offset
            samples.append((eye.tolist(), lap.tolist()))
        return samples


def sphere_to_cartesian(theta:float, phi:float) -> Tuple[float,float,float]:
    x = np.cos(theta) * np.sin(phi)
    y = np.cos(phi)
    z = np.sin(phi) * np.sin(theta)
    return [x,y,z]

def lookAt(eye, center, up):
    F = center - eye

    f = normalize(F)
    if abs(f[1]) > 0.99:
        f = normalize(up) * np.sign(f[1])
        u = np.array([0, 0, 1])
        s = np.cross(f, u)
    else:
        s = np.cross(f, normalize(up))
        u = np.cross(normalize(s), f)
    M = np.eye(4)
    M[0:3, 0] = s
    M[0:3, 1] = u
    M[0:3, 2] = -f

    T = np.eye(4)
    T[3, 0:3] = -eye
    return M @ T


def normalize(vec):
    return vec / np.linalg.norm(vec)


if __name__ == "__main__":
    scene_filepath = "scene_datasets/habitat-test-scene/apartment_1.glb"
    extractor = ImageExtractor(
        scene_filepath,
        img_size=(512, 512),
        output=["rgba", "depth"],
        pose_extractor_name="sphere_sampler",
        shuffle=False,
    )
    index = 2
    img = extractor.create_panorama(index)
    imageio.imwrite("test.png", img["rgba"])

    extractor.close()
