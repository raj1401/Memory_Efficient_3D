from plyfile import PlyData, PlyElement
import numpy as np


def get_matrix_of_gaussians(point_cloud_path):
    """
    This function returns an N times 14 matrix of gaussians for a given point cloud.
    N is the number of points in the point cloud. The 14 features are:
    x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z
    """
    ply_data = PlyData.read(point_cloud_path)
    num_gaussians = len(ply_data.elements[0].data)
    gaussians = np.zeros((num_gaussians, 14))

    vert = ply_data["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )

    sorted_indices = np.argsort(vert["opacity"])

    for i in sorted_indices:
        v = ply_data["vertex"][i]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scales = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )

        gaussians[i] = np.concatenate(
            [position, color, scales, rot], axis=0
        )
    
    return gaussians


def convert_gaussians_to_point_cloud(gaussians, output_path):
    """
    Converts an N x 14 matrix of gaussians into a PLY point cloud with the specified properties.
    
    Each row in `gaussians` should have the following 14 elements:
    x, y, z, r, g, b, opacity, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z
    
    Saves the output as a PLY file.
    """
    num_points = gaussians.shape[0]
    
    # Define the data structure with required fields
    point_cloud_data = np.zeros(
        num_points,
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('f_rest_0', 'f4'), ('f_rest_1', 'f4'), ('f_rest_2', 'f4'),
            ('f_rest_3', 'f4'), ('f_rest_4', 'f4'), ('f_rest_5', 'f4'),
            ('f_rest_6', 'f4'), ('f_rest_7', 'f4'), ('f_rest_8', 'f4'),
            ('f_rest_9', 'f4'), ('f_rest_10', 'f4'), ('f_rest_11', 'f4'),
            ('f_rest_12', 'f4'), ('f_rest_13', 'f4'), ('f_rest_14', 'f4'),
            ('f_rest_15', 'f4'), ('f_rest_16', 'f4'), ('f_rest_17', 'f4'),
            ('f_rest_18', 'f4'), ('f_rest_19', 'f4'), ('f_rest_20', 'f4'),
            ('f_rest_21', 'f4'), ('f_rest_22', 'f4'), ('f_rest_23', 'f4'),
            ('f_rest_24', 'f4'), ('f_rest_25', 'f4'), ('f_rest_26', 'f4'),
            ('f_rest_27', 'f4'), ('f_rest_28', 'f4'), ('f_rest_29', 'f4'),
            ('f_rest_30', 'f4'), ('f_rest_31', 'f4'), ('f_rest_32', 'f4'),
            ('f_rest_33', 'f4'), ('f_rest_34', 'f4'), ('f_rest_35', 'f4'),
            ('f_rest_36', 'f4'), ('f_rest_37', 'f4'), ('f_rest_38', 'f4'),
            ('f_rest_39', 'f4'), ('f_rest_40', 'f4'), ('f_rest_41', 'f4'),
            ('f_rest_42', 'f4'), ('f_rest_43', 'f4'), ('f_rest_44', 'f4'),
            ('opacity', 'f4'), 
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
        ]
    )
    
    SH_C0 = 0.28209479177387814  # Spherical Harmonic constant for color channels

    # Populate the point cloud data
    for i, gaussian in enumerate(gaussians):
        # Unpack gaussian parameters
        x, y, z = gaussian[0:3]
        r, g, b, opacity = gaussian[3:7]
        scale_x, scale_y, scale_z = gaussian[7:10]
        rot_w, rot_x, rot_y, rot_z = gaussian[10:14]

        point_cloud_data[i]['x'] = x
        point_cloud_data[i]['y'] = y
        point_cloud_data[i]['z'] = z
        point_cloud_data[i]['nx'] = 0.0
        point_cloud_data[i]['ny'] = 0.0
        point_cloud_data[i]['nz'] = 1.0

        # Convert color to spherical harmonics
        point_cloud_data[i]['f_dc_0'] = (r - 0.5) / SH_C0
        point_cloud_data[i]['f_dc_1'] = (g - 0.5) / SH_C0
        point_cloud_data[i]['f_dc_2'] = (b - 0.5) / SH_C0

        # Placeholder for SH coefficients `f_rest_0` to `f_rest_44`
        for k in range(45):
            point_cloud_data[i]['f_rest_' + str(k)] = 0.0

        # Set opacity
        point_cloud_data[i]['opacity'] = np.log(opacity / (1 - opacity)) if opacity < 1 else np.inf

        # Scales
        point_cloud_data[i]['scale_0'] = np.log(scale_x)
        point_cloud_data[i]['scale_1'] = np.log(scale_y)
        point_cloud_data[i]['scale_2'] = np.log(scale_z)

        # Quaternion rotation
        point_cloud_data[i]['rot_0'] = rot_w
        point_cloud_data[i]['rot_1'] = rot_x
        point_cloud_data[i]['rot_2'] = rot_y
        point_cloud_data[i]['rot_3'] = rot_z

    vertex_element = PlyElement.describe(point_cloud_data, 'vertex')
    PlyData([vertex_element], text=False).write(output_path)
