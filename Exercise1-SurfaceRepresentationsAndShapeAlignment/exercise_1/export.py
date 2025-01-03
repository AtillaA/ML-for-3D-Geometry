"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """
    # ------------------------------------------------
    with open(path, 'w') as fh:
        # vertices starting with "v "
        for v in vertices:
            fh.write("v {} {} {}\n".format(*v))

        # faces starting with "f "   
        for f in faces:
            fh.write("f {} {} {}\n".format(*(f + 1)))
    # ------------------------------------------------

def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ------------------------------------------------
    with open(path, 'w') as fh:
        for v in pointcloud:
            fh.write("v {} {} {}\n".format(*v))
    # ------------------------------------------------
