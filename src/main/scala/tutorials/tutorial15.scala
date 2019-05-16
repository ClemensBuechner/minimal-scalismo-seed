package tutorials

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry._
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RigidTransformationSpace, RotationTransform, TranslationTransform}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Memoize

object tutorial15 {

  def main(args: Array[String]): Unit = {

    implicit val rng = scalismo.utils.Random(42)
    scalismo.initialize()

    val ui = ScalismoUI()

    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("datasets/bfm.h5")).get

    val modelGroup = ui.createGroup("model")
    val modelView = ui.show(modelGroup, model, "model")
    modelView.meshView.opacity = 0.5

    val modelLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/modelLM_mcmc.json")).get
    val modelLmViews = ui.show(modelGroup, modelLms, "modelLandmarks")
    modelLmViews.foreach(lmView => lmView.color = java.awt.Color.BLUE)

    val targetGroup = ui.createGroup("target")

    val targetLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("datasets/targetLM_mcmc.json")).get
    val targetLmViews = ui.show(targetGroup, targetLms, "targetLandmarks")
    modelLmViews.foreach(lmView => lmView.color = java.awt.Color.RED)

    val modelLmIds =  modelLms.map(l => model.mean.pointSet.pointId(l.point).get)
    val targetPoints = targetLms.map(l => l.point)

    val landmarkNoiseVariance = 9.0
    val uncertainty = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance
    )

    val correspondences = modelLmIds.zip(targetPoints).map(modelIdWithTargetPoint => {
      val (modelId, targetPoint) =  modelIdWithTargetPoint
      (modelId, targetPoint, uncertainty)
    })

    case class Parameters(translationParameters: EuclideanVector[_3D],
                          rotationParameters: (Double, Double, Double),
                          modelCoefficients: DenseVector[Double])

    case class Sample(generatedBy : String, parameters : Parameters, rotationCenter: Point[_3D]) {
      def poseTransformation : RigidTransformation[_3D] = {

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

    // stop

    case class PriorEvaluator(model: StatisticalMeshModel)
      extends DistributionEvaluator[Sample] {

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

    case class SimpleCorrespondenceEvaluator(model: StatisticalMeshModel,
                                             correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
      extends DistributionEvaluator[Sample] {

      override def logValue(sample: Sample): Double = {

        val currModelInstance = model.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)

        val likelihoods = correspondences.map( correspondence => {
          val (id, targetPoint, uncertainty) = correspondence
          val modelInstancePoint = currModelInstance.pointSet.point(id)
          val observedDeformation = targetPoint - modelInstancePoint

          uncertainty.logpdf(observedDeformation.toBreezeVector)
        })


        val loglikelihood = likelihoods.sum
        loglikelihood
      }
    }

  }
}
