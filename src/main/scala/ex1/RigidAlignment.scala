package ex1

import java.awt.Color
import java.io.File

import scalismo.geometry._
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.mesh.TriangleMesh
import scalismo.registration.{LandmarkRegistration, RigidTransformation}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random


object RigidAlignment {

  def main(args: Array[String]) {

    scalismo.initialize()
    implicit val rng: Random = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val femoraGroup = ui.createGroup("femora")

    val refMesh: TriangleMesh[_3D] = MeshIO.readMesh(new java.io.File("datasets/femur.stl")).get
    val refLM: Seq[Landmark[_3D]] = LandmarkIO.readLandmarksJson[_3D](new File("datasets/femur" +
      ".json")).get

    val refMeshView = ui.show(femoraGroup, refMesh, "Reference Femur")
    refMeshView.color = Color.ORANGE


    for (x <- 0 to 49) {
      val femurMesh: TriangleMesh[_3D] = MeshIO.readMesh(new File("data/femora/meshes/" + x + "" +
        ".stl")).get
      val femurLM = LandmarkIO.readLandmarksJson[_3D](new File("data/femora/landmarks/" + x + "" +
        ".json")).get
      val lmIds = femurLM.map { lm => femurMesh.pointSet.findClosestPoint(lm.point).id }

      val bestTransform: RigidTransformation[_3D] = LandmarkRegistration
        .rigid3DLandmarkRegistration(femurLM, refLM, center = Point(0, 0, 0))
      val alignedFemur = femurMesh.transform(bestTransform)
      val alignedLM: Seq[Landmark[_3D]] = lmIds.map { id =>
        Landmark(id.toString, alignedFemur
          .pointSet.point(id))
      }

      MeshIO.writeMesh(alignedFemur, new File("data/femora/aligned/" + x + ".stl"))
      LandmarkIO.writeLandmarksJson[_3D](alignedLM, new File("data/femora/alignedLandmarks/" + x
        + ".json"))

      val femurMeshView = ui.show(femoraGroup, alignedFemur, "Femur" + x)
    }
  }
}
