package segmentationProject

import java.awt.Color
import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry._
import scalismo.io._
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel.asm.{ActiveShapeModel, PreprocessedImage}
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Memoize

import scala.io.StdIn


object secondTry {

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

    val model = asm.statisticalModel

    val tests = Array(4, 14, 23, 25, 30)
    val targets = Array(1, 9, 10, 13, 37)

    tests.foreach { i: Int =>

      val image = ImageIO.read3DScalarImage[Short](new File(dataDir + "test/" + i + ".nii")).get
        .map(_.toFloat)
      val reference = MeshIO.readMesh(new File(dataDir + "test/" + i + ".stl")).get

      val testGroup = ui.createGroup("test_" + i)
      val imgView = ui.show(testGroup, image, "image")
      val refView = ui.show(testGroup, reference, "reference")
      refView.color = Color.GREEN

      val preprocessedImage = asm.preprocessor(image)

      val likelihoodEvaluator = CachedEvaluator(ActiveShapeModelEvaluator(model, asm,
        preprocessedImage))
      val priorEvaluator = CachedEvaluator(PriorEvaluator(model))

      val posteriorEvaluator = ProductEvaluator(priorEvaluator, likelihoodEvaluator)

      val shapeUpdateProposal = ShapeUpdateProposal(model.rank, 0.1)
      val rotationUpdateProposal = RotationUpdateProposal(0.01)
      val translationUpdateProposal = TranslationUpdateProposal(1.0)
      val generator = MixtureProposal.fromProposalsWithTransition(
        (0.6, shapeUpdateProposal), (0.2, rotationUpdateProposal), (0.2, translationUpdateProposal)
      )

      val initialParameters = Parameters(
        EuclideanVector(0, 0, 0),
        (0.0, 0.0, 0.0),
        DenseVector.zeros[Double](model.rank)
      )

      val initialSample = Sample("initial", initialParameters, computeCenterOfMass(model.mean))

      val chain = MetropolisHastings(generator, posteriorEvaluator)
      val logger = new Logger()
      val mhIterator = chain.iterator(initialSample, logger)

      val samplingIterator = for ((sample, iteration) <- mhIterator.zipWithIndex) yield {
        println("iteration " + iteration)
        if (iteration % 300 == 0) {
          modelView.shapeModelTransformationView.shapeTransformationView.coefficients = sample
            .parameters.modelCoefficients
          modelView.shapeModelTransformationView.poseTransformationView.transformation = sample
            .poseTransformation
          val dist = scalismo.mesh.MeshMetrics.avgDistance(model.instance(sample.parameters.modelCoefficients).transform(sample
            .poseTransformation), reference)
          val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(model.instance(sample.parameters.modelCoefficients).transform(sample
            .poseTransformation), reference)
          println("Average Distance: " + dist)
          println("Hausdorff Distance: " + hausDist)
        }
        sample
      }

      val samples = samplingIterator.drop(1000).take(2000).toIndexedSeq


      println(logger.acceptanceRatios())
      // Map(RotationUpdateProposal (0.01) -> 0.6971894832275612, TranlationUpdateProposal (1.0) ->
      // 0.5043859649122807, ShapeUpdateProposal (0.1) -> 0.7907262398280362)

      val bestSample = samples.maxBy(posteriorEvaluator.logValue)
      val bestFit = model.instance(bestSample.parameters.modelCoefficients).transform(bestSample
        .poseTransformation)
      val resultGroup = ui.createGroup("result")
      ui.show(resultGroup, bestFit, "best fit")

      val dist = scalismo.mesh.MeshMetrics.avgDistance(bestFit, reference)
      val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(bestFit, reference)
      println("Average Distance: " + dist)
      println("Hausdorff Distance: " + hausDist)

      StdIn.readLine("Show next fit.")
    }

    //    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model,
    //      correspondences)
    //
    //    for ((id, _, _) <- newCorrespondences) {
    //      val meanPointPosition = computeMean(marginalizedModel, id)
    //      println(s"expected position for point at id $id  = $meanPointPosition")
    //      val cov = computeCovarianceFromSamples(marginalizedModel, id, meanPointPosition)
    //      println(s"posterior variance computed  for point at id (shape and pose) $id  = ${cov
    //      (0, 0)
    //      }, ${cov(1, 1)}, ${cov(2, 2)}")
    //    }
    // expected position for point at id PointId(0)  = Point3D(148.10137201095068,
    // -7.151235826279037,291.3929897650882)
    // posterior variance computed  for point at id (shape and pose) PointId(0)  =
    // 4.254464918957609, 3.939366851939424, 3.32790002246177
    // expected position for point at id PointId(1)  = Point3D(142.24949688108646,
    // -4.4064122402505745,266.34241057439533)
    // posterior variance computed  for point at id (shape and pose) PointId(1)  =
    // 2.918968756161676, 2.159129807301085, 2.915190937982501
    // expected position for point at id PointId(2)  = Point3D(141.3051915247215,
    // -4.2897512164448095,226.94538458273826)
    // posterior variance computed  for point at id (shape and pose) PointId(2)  =
    // 2.203212589533782, 1.9443312683620244, 3.558990457373749
    // expected position for point at id PointId(3)  = Point3D(143.33662067133574,
    // -4.885817375124605,201.61254954128444)
    // posterior variance computed  for point at id (shape and pose) PointId(3)  =
    // 4.211176241740335, 3.5805967352101438, 3.8214742761066214
    // expected position for point at id PointId(4)  = Point3D(102.9147937651206,
    // 32.77448813851955,248.52670379739345)
    // posterior variance computed  for point at id (shape and pose) PointId(4)  =
    // 3.670091389108584, 4.998066354925429, 3.3640625099189827
    // expected position for point at id PointId(5)  = Point3D(137.63389362946234,
    // 101.3277589516202,248.64874726494665)
    // posterior variance computed  for point at id (shape and pose) PointId(5)  =
    // 8.479207875196545, 6.924214739937887, 5.29092402298062

    //    val posteriorFixedPose = model.posterior(correspondences.toIndexedSeq)
    //
    //    for ((id, _, _) <- newCorrespondences) {
    //      val cov = posteriorFixedPose.cov(id, id)
    //      println(s"posterior variance computed by analytic posterior (shape only) for point
    //      with id " +
    //        s"$id = ${cov(0, 0)}, ${cov(1, 1)}, ${cov(2, 2)}")
    //    }
    // posterior variance computed by analytic posterior (shape only) for point with id PointId
    // (0) = 1.7493438462816924, 2.3773798978463545, 2.4529203603071132
    // posterior variance computed by analytic posterior (shape only) for point with id PointId
    // (1) = 1.7425234299448356, 2.347569081897583, 2.484181513654568
    // posterior variance computed by analytic posterior (shape only) for point with id PointId
    // (2) = 1.728312680395977, 2.311983615096077, 2.4760881301654973
    // posterior variance computed by analytic posterior (shape only) for point with id PointId
    // (3) = 1.7362509228394625, 2.3504558671936087, 2.4497130086130414
    // posterior variance computed by analytic posterior (shape only) for point with id PointId
    // (4) = 1.74656771644886, 2.315887303162952, 2.494671632799402
    // posterior variance computed by analytic posterior (shape only) for point with id PointId
    // (5) = 1.7511296018408746, 2.2802845887546255, 2.505733093422708
  }

  def computeCenterOfMass(mesh: TriangleMesh[_3D]): Point[_3D] = {
    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector *
      normFactor)
  }

  //  def computeMean(model: StatisticalMeshModel, id: PointId, samples: Seq[Sample]): Point[_3D]
  //  = {
  //    var mean = EuclideanVector(0, 0, 0)
  //    for (sample <- samples) yield {
  //      val modelInstance = model.instance(sample.parameters.modelCoefficients)
  //      val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet
  //      .point(id)
  //      mean += pointForInstance.toVector
  //    }
  //    (mean * 1.0 / samples.size).toPoint
  //  }
  //
  //  def computeCovarianceFromSamples(model: StatisticalMeshModel, id: PointId, mean: Point[_3D],
  //                                   samples: Seq[Sample]): SquareMatrix[_3D] = {
  //
  //    var cov = SquareMatrix.zeros[_3D]
  //    for (sample <- samples) yield {
  //      val modelInstance = model.instance(sample.parameters.modelCoefficients)
  //      val pointForInstance = modelInstance.transform(sample.poseTransformation).pointSet
  //      .point(id)
  //      val v = pointForInstance - mean
  //      cov += v.outer(v)
  //    }
  //    cov * (1.0 / samples.size)
  //  }
}

case class Parameters(translationParameters: EuclideanVector[_3D],
                      rotationParameters: (Double, Double, Double),
                      modelCoefficients: DenseVector[Double])

case class Sample(generatedBy: String, parameters: Parameters, rotationCenter: Point[_3D]) {
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

case class ActiveShapeModelEvaluator(model: StatisticalMeshModel, asm: ActiveShapeModel,
                                     preprocessedImage: PreprocessedImage) extends
  DistributionEvaluator[Sample] {

  override def logValue(sample: Sample): Double = {

    val ids = asm.profiles.ids
    val mesh = model.instance(sample.parameters.modelCoefficients).transform(sample
      .poseTransformation)

    val likelihoods = for (id <- ids) yield {
      val profile = asm.profiles(id)
      val profilePointOnMesh = mesh.pointSet.point(profile.pointId)
      val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh,
        profile.pointId).get
      profile.distribution.logpdf(featureAtPoint)
    }
    likelihoods.sum
  }
}

case class CorrespondenceEvaluator(model: StatisticalMeshModel, correspondences: Seq[
  (PointId, Point[_3D], MultivariateNormalDistribution)]) extends
  DistributionEvaluator[Sample] {

  val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model,
    correspondences)

  def marginalizeModelForCorrespondences(model: StatisticalMeshModel, correspondences: Seq[
    (PointId, Point[_3D], MultivariateNormalDistribution)]): (StatisticalMeshModel, Seq[
    (PointId, Point[_3D], MultivariateNormalDistribution)]) = {

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
}

case class CachedEvaluator[A](evaluator: DistributionEvaluator[A]) extends
  DistributionEvaluator[A] {
  val memoizedLogValue = Memoize(evaluator.logValue, 10)

  override def logValue(sample: A): Double = {
    memoizedLogValue(sample)
  }
}

case class ShapeUpdateProposal(paramVectorSize: Int, stddev: Double) extends
  ProposalGenerator[Sample] with TransitionProbability[Sample] {

  val perturbationDistr = new MultivariateNormalDistribution(
    DenseVector.zeros(paramVectorSize),
    DenseMatrix.eye[Double](paramVectorSize) * stddev * stddev
  )

  implicit val rng = scalismo.utils.Random(42)

  override def propose(sample: Sample): Sample = {
    val newParameters = sample.parameters.copy(modelCoefficients = sample.parameters
      .modelCoefficients + perturbationDistr.sample)
    sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample) = {
    val residual = to.parameters.modelCoefficients - from.parameters.modelCoefficients
    perturbationDistr.logpdf(residual)
  }
}

case class RotationUpdateProposal(stddev: Double) extends
  ProposalGenerator[Sample] with TransitionProbability[Sample] {

  val perturbationDistr = new MultivariateNormalDistribution(
    DenseVector.zeros[Double](3),
    DenseMatrix.eye[Double](3) * stddev * stddev)

  implicit val rng = scalismo.utils.Random(42)

  def propose(sample: Sample): Sample = {
    val perturbation = perturbationDistr.sample
    val newRotationParameters = (
      sample.parameters.rotationParameters._1 + perturbation(0),
      sample.parameters.rotationParameters._2 + perturbation(1),
      sample.parameters.rotationParameters._3 + perturbation(2)
    )
    val newParameters = sample.parameters.copy(rotationParameters = newRotationParameters)
    sample.copy(generatedBy = s"RotationUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample) = {
    val residual = DenseVector(
      to.parameters.rotationParameters._1 - from.parameters.rotationParameters._1,
      to.parameters.rotationParameters._2 - from.parameters.rotationParameters._2,
      to.parameters.rotationParameters._3 - from.parameters.rotationParameters._3
    )
    perturbationDistr.logpdf(residual)
  }
}

case class TranslationUpdateProposal(stddev: Double) extends
  ProposalGenerator[Sample] with TransitionProbability[Sample] {

  val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3),
    DenseMatrix.eye[Double](3) * stddev * stddev)

  implicit val rng = scalismo.utils.Random(42)

  def propose(sample: Sample): Sample = {
    val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector
      .fromBreezeVector(perturbationDistr.sample())
    val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
    sample.copy(generatedBy = s"TranlationUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample) = {
    val residual = to.parameters.translationParameters - from.parameters.translationParameters
    perturbationDistr.logpdf(residual.toBreezeVector)
  }
}

class Logger extends AcceptRejectLogger[Sample] {
  private val numAccepted = collection.mutable.Map[String, Int]()
  private val numRejected = collection.mutable.Map[String, Int]()

  override def accept(current: Sample,
                      sample: Sample,
                      generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]
                     ): Unit = {
    val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
    numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
  }

  override def reject(current: Sample,
                      sample: Sample,
                      generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]
                     ): Unit = {
    val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
    numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
  }


  def acceptanceRatios(): Map[String, Double] = {
    val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
    val acceptanceRatios = for (generatorName <- generatorNames) yield {
      val total = (numAccepted.getOrElse(generatorName, 0)
        + numRejected.getOrElse(generatorName, 0)).toDouble
      (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
    }
    acceptanceRatios.toMap
  }
}
