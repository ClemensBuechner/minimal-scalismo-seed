package com.example

import java.awt.Color

import scalismo.io.{LandmarkIO, MeshIO, StatismoIO}
import java.io.File

import scalismo.ui.api.ScalismoUI
import scalismo.geometry._
import scalismo.common._
import scalismo.ui.api._
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RotationTransform, Transformation, TranslationTransform}
import scalismo.registration.LandmarkRegistration


object ExampleApp {

  def main(args: Array[String]) {

    // required to initialize native libraries (VTK, HDF5 ..)
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    // Your application code goes below here. Below is a dummy application that reads a mesh and displays it

    // create a visualization window
    val ui = ScalismoUI()

    val femoraGroup = ui.createGroup("femora")

    val refMesh: TriangleMesh[_3D] = MeshIO.readMesh(new java.io.File("datasets/femur.stl")).get

    for (x <- 0 to 49) {
      val femurMesh: TriangleMesh[_3D] = MeshIO.readMesh(new java.io.File("data/femora/meshes/1.stl")).get
      val femurLM = LandmarkIO.readLandmarksJson[_3D](new File("data/femora/landmarks/1.json")).get
      val refLM = LandmarkIO.readLandmarksJson[_3D](new File("datasets/femur.json")).get
      val bestTransform: RigidTransformation[_3D] = LandmarkRegistration.rigid3DLandmarkRegistration(femurLM, refLM, center = Point(0, 0, 0))
      val alignedFemur = femurMesh.transform(bestTransform)
      MeshIO.writeMesh(alignedFemur, new File("data/femora/aligned/" + x + ".stl"))
    }

    val translation = TranslationTransform[_3D](EuclideanVector(100, 0, 0))
    val rotationCenter = Point(0.0, 0.0, 0.0)
    val rotation: RotationTransform[_3D] = RotationTransform(0f, 3.14f, 0f, rotationCenter)
    val rigidTransform2: RigidTransformation[_3D] = RigidTransformation[_3D](translation, rotation)

    val refMeshView = ui.show(femoraGroup, refMesh, "Reference Femur")
    refMeshView.color = Color.ORANGE

    //    val femurMeshView = ui.show(femoraGroup, femurMesh, "Femur")
    //    femurMeshView.color = Color.BLUE
    //
    //    val alignedFemurView = ui.show(femoraGroup, alignedFemur, "alignedPaola")
    //    alignedFemurView.color = Color.GREEN


        val bfmModel = StatismoIO.readStatismoMeshModel(new File("datasets/bfm.h5")).get
        ui.show(bfmModel, "model")
        val paolaMesh : TriangleMesh[_3D] = MeshIO.readMesh(new java.io.File("datasets/Paola.stl")).get

        val face0 = bfmModel.sample();
        ui.show(face0, "face0")

        val face1 = bfmModel.sample();
        ui.show(face1, "face1")

        val face2 = bfmModel.sample();
        ui.show(face2, "face2")

        val face3 = bfmModel.sample();
        ui.show(face3, "face3")

        val face4 = bfmModel.sample();
        ui.show(face4, "face4")

        val face5 = bfmModel.sample();
        ui.show(face5, "face5")

        val face6 = bfmModel.sample();
        ui.show(face6, "face6")

        val face7 = bfmModel.sample();
        ui.show(face7, "face7")

        val face8 = bfmModel.sample();
        ui.show(face8, "face8")

        val face9 = bfmModel.sample();
        ui.show(face9, "face9")

              val faceLM = face0.cells
              val refLM = paolaMesh.cells
              val bestTransform : RigidTransformation[_3D] = LandmarkRegistration.rigid3DLandmarkRegistration(faceLM, refLM, center = Point(0, 0, 0))
              val alignedFace = face0.transform(bestTransform)
  }
}
