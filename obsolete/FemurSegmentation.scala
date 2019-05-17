package segmentationProject

import java.awt.Color
import java.io.File

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector, Point3D, _3D}
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO, StatisticalModelIO}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.sampling.DistributionEvaluator
import scalismo.statisticalmodel.asm._
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Memoize

object FemurSegmentation {

  def main(args: Array[String]): Unit = {

    implicit val rng = scalismo.utils.Random(42)
    scalismo.initialize()
    val ui = ScalismoUI()

    val dataDir = "data/handedData/"

    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File(dataDir +
      "femur-asm.h5")).get
    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, asm.statisticalModel, "shapeModel")

    val profiles = asm.profiles
    profiles.foreach(profile => {
      val pointId = profile.pointId
      val distribution = profile.distribution
    })
    println("Loaded model.")

    val searchPointSampler = NormalDirectionSearchPointSampler(numberOfPoints = 100,
      searchDistance = 3)
    val config = FittingConfiguration(featureDistanceThreshold = 3, pointDistanceThreshold = 5,
      modelCoefficientBounds = 3)


    val tests = Array(4, 14, 23, 25, 30)
    val targets = Array(1, 9, 10, 13, 37)
    tests.foreach { i: Int =>
      val image = ImageIO.read3DScalarImage[Short](new File(dataDir + "test/" + i + ".nii")).get
        .map(_.toFloat)

      val reference = MeshIO.readMesh(new File(dataDir + "test/" + i + ".stl")).get

      val testGroup = ui.createGroup("test_" + i)
      val imgView = ui.show(testGroup, image, "image_" + i)
      val refView = ui.show(testGroup, reference, "reference")
      refView.color = Color.GREEN

      val preprocessedImage = asm.preprocessor(image)

      val modelBoundingBox = asm.statisticalModel.referenceMesh.boundingBox
      val rotationCenter = modelBoundingBox.origin + modelBoundingBox.extent * 0.5

      val translationTransformation = TranslationTransform(EuclideanVector(0, 0, 0))
      val rotationTransformation = RotationTransform(0, 0, 0, rotationCenter)
      val initialRigidTransformation = RigidTransformation(translationTransformation,
        rotationTransformation)
      val initialModelCoefficients = DenseVector.zeros[Double](asm.statisticalModel.rank)
      val initialTransformation = ModelTransformations(initialModelCoefficients,
        initialRigidTransformation)

      val numberOfIterations = 20
      val asmIterator = asm.fitIterator(image, searchPointSampler, numberOfIterations, config,
        initialTransformation)

      val asmIteratorWithVisualization = asmIterator.map(it => {
        it match {
          case scala.util.Success(iterationResult) => {
            modelView.shapeModelTransformationView.poseTransformationView.transformation =
              iterationResult.transformations.rigidTransform
            modelView.shapeModelTransformationView.shapeTransformationView.coefficients =
              iterationResult.transformations.coefficients
          }
          case scala.util.Failure(error) => System.out.println(error.getMessage)
        }
        it
      })

      val result = asmIteratorWithVisualization.toIndexedSeq.last
      val finalMesh = result.get.mesh

      val testView = ui.show(testGroup, finalMesh, "testMesh")
    }

    //    val targetCTs = targets.map { i: Int =>
    //      ImageIO.read3DScalarImage[Short](new File(dataDir + "targets/" + i + ".nii")).get.map(_
    //        .toFloat)
    //    }
    //    println("Loaded targets.")
    //
    //
    //
    //    val preprocessedTests = testCTs.map { img =>
    //      asm.preprocessor(img)
    //    }
    //    val preprocessedTargets = targetCTs.map { img =>
    //      asm.preprocessor(img)
    //    }
  }
}

case class Parameters(translationParameters: EuclideanVector[_3D],
                      rotationParameters: (Double, Double, Double),
                      modelCoefficients: DenseVector[Double])

case class Sample(generatedBy: String, parameters: Parameters, rotationCenter: Point3D) {
  def poseTransformation: RigidTransformation[_3D] = {

    val translation = TranslationTransform(parameters.translationParameters)
    val rotation = RotationTransform(
      parameters.rotationParameters._1,
      parameters.rotationParameters._2,
      parameters.rotationParameters._3,
      rotationCenter
    )
    RigidTransformation(translation, rotation)
  }
}

case class PriorEvaluator(model: StatisticalMeshModel) extends DistributionEvaluator[Sample] {

  val translationPrior = breeze.stats.distributions.Gaussian(0.0, 5.0)
  val rotationPrior = breeze.stats.distributions.Gaussian(0, 0.1)

  override def logValue(sample: Sample): Double = {
    model.gp.logpdf(sample.parameters.modelCoefficients) +
      translationPrior.logPdf(sample.parameters.translationParameters.x) +
      translationPrior.logPdf(sample.parameters.translationParameters.y) +
      translationPrior.logPdf(sample.parameters.translationParameters.z) +
      rotationPrior.logPdf(sample.parameters.rotationParameters._1) +
      rotationPrior.logPdf(sample.parameters.rotationParameters._2) +
      rotationPrior.logPdf(sample.parameters.rotationParameters._3)
  }
}

//case class ActiveShapeModelEvaluator(model: StatisticalMeshModel, asm: ActiveShapeModel,
//                                     mesh: TriangleMesh[_3D],
//                                     preprocessedImage: PreprocessedImage) extends
//  DistributionEvaluator[Sample] {
//
//  override def logValue(sample: Sample): Double = {
//
//      val ids = asm.profiles.ids
//
//      val likelihoods = for (id <- ids) yield {
//        val profile = asm.profiles(id)
//        val profilePointOnMesh = mesh.pointSet.point(profile.pointId)
//        val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh,
//          profile.pointId).get
//        profile.distribution.logpdf(featureAtPoint)
//      }
//      likelihoods.sum
//    }
//  }

  case class SimpleCorrespondenceEvaluator(model: StatisticalMeshModel, correspondences: Seq[
    (PointId, Point3D, MultivariateNormalDistribution)]) extends DistributionEvaluator[Sample] {

    override def logValue(sample: Sample): Double = {

      val currModelInstance = model.instance(sample.parameters.modelCoefficients).transform(sample
        .poseTransformation)

      val likelihoods = correspondences.map(correspondence => {
        val (id, targetPoint, uncertainty) = correspondence
        val modelInstancePoint = currModelInstance.pointSet.point(id)
        val observedDeformation = targetPoint - modelInstancePoint

        uncertainty.logpdf(observedDeformation.toBreezeVector)
      })


      val loglikelihood = likelihoods.sum
      loglikelihood
    }
  }

  case class CorrespondenceEvaluator(model: StatisticalMeshModel, correspondences: Seq[(PointId,
    Point3D, MultivariateNormalDistribution)]) extends DistributionEvaluator[Sample] {

    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model,
      correspondences)

    override def logValue(sample: Sample): Double = {

      val currModelInstance = marginalizedModel.instance(sample.parameters.modelCoefficients)
        .transform(sample.poseTransformation)

      val likelihoods = newCorrespondences.map(correspondence => {
        val (id, targetPoint, uncertainty) = correspondence
        val modelInstancePoint = currModelInstance.pointSet.point(id)
        val observedDeformation = targetPoint - modelInstancePoint

        uncertainty.logpdf(observedDeformation.toBreezeVector)
      })


      val loglikelihood = likelihoods.sum
      loglikelihood
    }

    def marginalizeModelForCorrespondences(model: StatisticalMeshModel, correspondences: Seq[
      (PointId, Point3D, MultivariateNormalDistribution)]): (StatisticalMeshModel, Seq[(PointId,
      Point3D, MultivariateNormalDistribution)]) = {

      val (modelIds, _, _) = correspondences.unzip3
      val marginalizedModel = model.marginal(modelIds.toIndexedSeq)
      val newCorrespondences = correspondences.map(idWithTargetPoint => {
        val (id, targetPoint, uncertainty) = idWithTargetPoint
        val modelPoint = model.referenceMesh.pointSet.point(id)
        val newId = marginalizedModel.referenceMesh.pointSet.findClosestPoint(modelPoint).id
        (newId, targetPoint, uncertainty)
      })
      (marginalizedModel, newCorrespondences)
    }
  }

  case class CachedEvaluator[A](evaluator: DistributionEvaluator[A]) extends
    DistributionEvaluator[A] {
    val memoizedLogValue = Memoize(evaluator.logValue, 10)

    override def logValue(sample: A): Double = {
      memoizedLogValue(sample)
    }
  }

