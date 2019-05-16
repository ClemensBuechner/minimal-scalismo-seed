package segmentationProject

import java.io.File

import breeze.linalg.DenseVector
import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector, Point3D, _3D}
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO, StatisticalModelIO}
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.sampling.DistributionEvaluator
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
    profiles.foreach( profile => {
      val pointId = profile.pointId
      val distribution = profile.distribution
    })
    println("Loaded model.")

    val tests = Array(4, 14, 23, 25, 30)
    val targets = Array(1, 9, 10, 13, 37)
    val testCTs = tests.map { i: Int =>
      ImageIO.read3DScalarImage[Short](new File(dataDir + "test/" + i + ".nii"))
    }
    val preprocessedTests = testCTs.map { img =>
      asm.preprocessor(img)
    }
    val testReferences = tests.map { i: Int =>
      MeshIO.readMesh(new File(dataDir + "test/" + i + ".stl"))
    }
    println("Loaded tests.")

    val targetCTs = targets.map { i: Int =>
      ImageIO.read3DScalarImage[Short](new File(dataDir + "targets/" + i + ".nii"))
    }
    val preprocessedTargets = targetCTs.map { img =>
      asm.preprocessor(img)
    }
    println("Loaded targets.")


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

