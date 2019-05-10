package tutorials

import scalismo.geometry._
import scalismo.ui.api._
import scalismo.registration._
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.asm._
import scalismo.io.{ActiveShapeModelIO, ImageIO}
import breeze.linalg.{DenseVector}

object tutorial13 {

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val dataDir = "data/handedData/"
    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File(dataDir + "femur-asm.h5")).get

    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, asm.statisticalModel, "shapeModel")

    val profiles = asm.profiles
    profiles.foreach(profile => {
      val pointId = profile.pointId
      val distribution = profile.distribution
    })

    val image = ImageIO.read3DScalarImage[Short](new java.io.File(dataDir + "targets/1.nii")).get.map(_.toFloat)
    val targetGroup = ui.createGroup("target")

    val imageView = ui.show(targetGroup, image, "image")

    // TODO: UnsupportedClassVersionError: scalismo/image/BSplineCoefficients
    val preprocessedImage = asm.preprocessor(image)

    val point1 = image.domain.origin + EuclideanVector(10.0, 10.0, 10.0)
    val profile = asm.profiles.head
    val feature1 : DenseVector[Double] = asm.featureExtractor(preprocessedImage, point1, asm.statisticalModel.mean, profile.pointId).get

  }
}
