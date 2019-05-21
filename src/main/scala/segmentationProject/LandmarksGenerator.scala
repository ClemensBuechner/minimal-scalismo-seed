package segmentationProject

import java.awt.Color
import java.io.File

import scalismo.geometry._3D
import scalismo.image.DiscreteScalarImage
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO}
import scalismo.mesh.TriangleMesh3D
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.ui.api.ScalismoUI

import scala.io.StdIn

object LandmarksGenerator {

  def getLandmarksForModel(ui: ScalismoUI, model: StatisticalMeshModel) = {

    val modelGroup = ui.createGroup("model")
    val modelView = ui.show(modelGroup, model, "model")

    StdIn.readLine("Click landmarks in the model, save them, and press [Enter].")

    modelGroup.remove()
  }

  def getLandmarksForImage(ui: ScalismoUI, image: DiscreteScalarImage[_3D, Float],
                           reference: TriangleMesh3D) = {

    val imageGroup = ui.createGroup("image")
    val imageView = ui.show(imageGroup, image, "image")

    if (reference != null) {
      val refView = ui.show(imageGroup, reference, "reference")
      refView.color = Color.GREEN
    }

    StdIn.readLine("Click landmarks in the CT image, save them, and press [Enter].")

    imageGroup.remove()
  }

  def main(args: Array[String]): Unit = {

    implicit val rng = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val dataDir = "data/handedData/"

    val asm = ActiveShapeModelIO.readActiveShapeModel(new File(dataDir + "femur-asm.h5")).get
    getLandmarksForModel(ui, asm.statisticalModel)

    val tests = Array(4, 14, 23, 25, 30)
    val targets = Array(1, 9, 10, 13, 37)

    println("tests")
    tests.foreach { i: Int =>
      val image = ImageIO.read3DScalarImage[Short](new File(dataDir + "test/" + i + ".nii")).get
        .map(_.toFloat)
      val reference = MeshIO.readMesh(new File(dataDir + "test/" + i + ".stl")).get
      println("Loaded test_" + i)
      getLandmarksForImage(ui, image, reference)
    }

    println("targets")
    targets.foreach { i: Int =>
      val image = ImageIO.read3DScalarImage[Short](new File(dataDir + "targets/" + i + ".nii")).get
        .map(_.toFloat)
      println("Loaded target_" + i)
      getLandmarksForImage(ui, image, null)
    }
  }
}

//Ausrichtung: Head nach oben links
//A: Intercondyloid Fossa  von unten (unten mitte grube)
//B: Medial condyle von unten (unten links)
//C: Lateral condyle von unten (unten rechts)
//D: Hintere grube von hinten
//E: Medial condyle von vorne
//F: Lateral condyle von vorne
//G: Lesser Trochanter von vorne
//H: Fovea capitis von links oben
//I: Obturator iternus von oben
//J: Greater Trochanter von rechts

