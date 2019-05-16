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

    val point2 = image.domain.origin + EuclideanVector(20.0, 10.0, 10.0)
    val featureVec1 = asm.featureExtractor(preprocessedImage, point1, asm.statisticalModel.mean, profile.pointId).get
    val featureVec2 = asm.featureExtractor(preprocessedImage, point2, asm.statisticalModel.mean, profile.pointId).get

    val probabilityPoint1 = profile.distribution.logpdf(featureVec1)
    val probabilityPoint2 = profile.distribution.logpdf(featureVec2)

    val searchSampler = NormalDirectionSearchPointSampler(numberOfPoints = 100, searchDistance = 3)

    val config = FittingConfiguration(featureDistanceThreshold = 3, pointDistanceThreshold = 5, modelCoefficientBounds = 3)

    // make sure we rotate around a reasonable center point
    val modelBoundingBox = asm.statisticalModel.referenceMesh.boundingBox
    val rotationCenter = modelBoundingBox.origin + modelBoundingBox.extent * 0.5

    // we start with the identity transform
    val translationTransformation = TranslationTransform(EuclideanVector(0, 0, 0))
    val rotationTransformation = RotationTransform(0, 0, 0, rotationCenter)
    val initialRigidTransformation = RigidTransformation(translationTransformation, rotationTransformation)
    val initialModelCoefficients = DenseVector.zeros[Double](asm.statisticalModel.rank)
    val initialTransformation = ModelTransformations(initialModelCoefficients, initialRigidTransformation)

    val numberOfIterations = 20
    val asmIterator = asm.fitIterator(image, searchSampler, numberOfIterations, config, initialTransformation)

    val asmIteratorWithVisualization = asmIterator.map(it => {
      it match {
        case scala.util.Success(iterationResult) => {
          modelView.shapeModelTransformationView.poseTransformationView.transformation = iterationResult.transformations.rigidTransform
          modelView.shapeModelTransformationView.shapeTransformationView.coefficients = iterationResult.transformations.coefficients
        }
        case scala.util.Failure(error) => System.out.println(error.getMessage)
      }
      it
    })

    val result = asmIteratorWithVisualization.toIndexedSeq.last
    val finalMesh = result.get.mesh

    val finalMeshView = ui.show(targetGroup, finalMesh, "finalMesh")

    def likelihoodForMesh(asm : ActiveShapeModel, mesh : TriangleMesh[_3D], preprocessedImage: PreprocessedImage) : Double = {

      val ids = asm.profiles.ids

      val likelihoods = for (id <- ids) yield {
        val profile = asm.profiles(id)
        val profilePointOnMesh = mesh.pointSet.point(profile.pointId)
        val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh, profile.pointId).get
        profile.distribution.logpdf(featureAtPoint)
      }
      likelihoods.sum
    }

    val sampleMesh1 = asm.statisticalModel.sample
    val sampleMesh2 = asm.statisticalModel.sample
    println("Likelihood for mesh 1 = " + likelihoodForMesh(asm, sampleMesh1, preprocessedImage))
    println("Likelihood for mesh 2 = " + likelihoodForMesh(asm, sampleMesh2, preprocessedImage))

  }
}
