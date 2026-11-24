from pathlib import Path
import shutil
import subprocess

folder = Path.home() / "EECS4421/Project/aruco_board"

model = Path.home() / ".gazebo/models/aruco_board"
textures = model / "materials/textures"
scripts = model / "materials/scripts"

textures.mkdir(parents=True, exist_ok=True)
scripts.mkdir(parents=True, exist_ok=True)

shutil.copy(folder / "aruco_board.png", textures)

(scripts / "board.material").write_text("""
material Aruco/Board
{
    technique
    {
        pass
        {
            lighting off
            texture_unit
            {
                texture aruco_board.png
                filtering none
            }
        }
    }
}
""")

(model / "model.config").write_text("""
<model>
    <name>aruco_board</name>
    <version>1.0</version>
    <sdf version="1.6">model.sdf</sdf>
</model>
""")

(model / "model.sdf").write_text("""
<sdf version="1.6">
  <model name="aruco_board">
    <static>true</static>
    <link name="board_link">
      <visual name="marker_surface">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>0.40 0.30</size>
          </plane>
        </geometry>
        <material>
          <script>
            <uri>model://aruco_board/materials/scripts</uri>
            <uri>model://aruco_board/materials/textures</uri>
            <name>Aruco/Board</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
""")

subprocess.run([
    "ros2", "run", "gazebo_ros", "spawn_entity.py",
    "-entity", "aruco_board",
    "-file", str(model / "model.sdf"),
    "-x", "0.697",
    "-y", "-0.010",
    "-z", "0.913",
    "-R", "1.570796",
    "-P", "0",
    "-Y", "-1.570796"
], check=True)




