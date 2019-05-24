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
import scalismo.ui.api.{ScalismoUI, StatisticalMeshModelViewControls}
import scalismo.utils.{Memoize, Random}

import scala.io.StdIn


object secondTry {

  implicit val rng: Random = Random(42)

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    val ui = ScalismoUI()

    val dataDir = "data/handedData/"

    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File(dataDir +
      "femur-asm.h5")).get
    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, asm.statisticalModel, "shapeModel")

    val modelLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File
    ("data/landmarks/model_transformed.json")).get
    val modelLmViews = ui.show(modelGroup, modelLms, "modelLandmarks")
    modelLmViews.foreach(lmView => lmView.color = java.awt.Color.BLUE)

    val profiles = asm.profiles
    profiles.foreach(profile => {
      val pointId = profile.pointId
      val distribution = profile.distribution
    })
    println("Loaded model.")

    val tests = Array(/*4, 14, 23, 25,*/ 30)
    val targets = Array(1, 9, 10, 13, 37)

    tests.foreach { i: Int =>

      val model = asm.statisticalModel

      val image = ImageIO.read3DScalarImage[Short](new File(dataDir + "test/" + i + ".nii")).get
        .map(_.toFloat)
      val reference = MeshIO.readMesh(new File(dataDir + "test/" + i + ".stl")).get

      val testGroup = ui.createGroup("test_" + i)
      val imgView = ui.show(testGroup, image, "image")
      val refView = ui.show(testGroup, reference, "reference")
      refView.color = Color.GREEN

      val imgLms = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/landmarks/" + i +
        "_transformed.json")).get
      val imgLmViews = ui.show(testGroup, imgLms, "imgLandmarks")
      imgLmViews.foreach(lmView => lmView.color = java.awt.Color.RED)

      val modelLmIds = modelLms.map(l => model.mean.pointSet.pointId(l.point).get)
      val modelLmPoints = modelLmIds.map(id => model.mean.pointSet.point(id))
      val imgPoints = imgLms.map(l => l.point)


      val preprocessedImage = asm.preprocessor(image)

      val priorEvaluator = CachedEvaluator(PriorEvaluator(model))

      val likelihoodEvaluatorLM = getLandmarkLikelihoodEvaluator(model, modelLmIds, imgPoints)
      val posteriorEvaluatorLM = ProductEvaluator(priorEvaluator, likelihoodEvaluatorLM)

      val likelihoodEvaluatorASM = CachedEvaluator(ActiveShapeModelEvaluator(model, asm,
        preprocessedImage))
      val posteriorEvaluatorASM = ProductEvaluator(priorEvaluator, likelihoodEvaluatorASM, likelihoodEvaluatorLM)


      val shapeUpdateSmallProposal = ShapeUpdateProposal(model.rank, 0.01)
      val shapeUpdateMediumProposal = ShapeUpdateProposal(model.rank, 0.1)
      val shapeUpdateLargeProposal = ShapeUpdateProposal(model.rank, 1)
      val rotationUpdateProposal = RotationUpdateProposal(0.01)
      val translationUpdateProposal = TranslationUpdateProposal(1.0)

//      Using large=0.1, medium=0.01, small=0.001
//      3 chains, first landmarks, then twice ASM
//      5000 iterations each
//
//      test 4
//      Map(ShapeUpdateProposal (0.1) -> 0.1964465303385853, TranlationUpdateProposal (1.0) -> 0.20953660174613833, ShapeUpdateProposal (0.001) -> 0.009976057462090982, RotationUpdateProposal (0.01) -> 0.1327974276527331, ShapeUpdateProposal (0.01) -> 0.29502923976608186)
//      Average Distance: 0.7193388188196546
//      Hausdorff Distance: 6.058696500512402
//
//      test 14
//      Map(ShapeUpdateProposal (0.1) -> 0.19497260715436673, TranlationUpdateProposal (1.0) -> 0.22887208155212102, ShapeUpdateProposal (0.001) -> 0.017262143717382578, RotationUpdateProposal (0.01) -> 0.11076497057805469, ShapeUpdateProposal (0.01) -> 0.2781456953642384)
//      Average Distance: 0.6663277677851256
//      Hausdorff Distance: 4.475893011358805
//
//      test 23
//      Map(ShapeUpdateProposal (0.1) -> 0.1932383162081538, TranlationUpdateProposal (1.0) -> 0.21181938911022577, ShapeUpdateProposal (0.001) -> 0.011177644710578843, RotationUpdateProposal (0.01) -> 0.141785957736878, ShapeUpdateProposal (0.01) -> 0.3148200623406064)
//      Average Distance: 0.693717601538942
//      Hausdorff Distance: 3.234064625525209
//
//      test 25
//      Map(ShapeUpdateProposal (0.1) -> 0.19501515661839003, TranlationUpdateProposal (1.0) -> 0.21506442021803765, ShapeUpdateProposal (0.001) -> 0.019398258115597783, RotationUpdateProposal (0.01) -> 0.15933694181326116, ShapeUpdateProposal (0.01) -> 0.30463199772662686)
//      Average Distance: 0.6926581108340492
//      Hausdorff Distance: 3.3740502761280595
//
//      test 30
//      Map(ShapeUpdateProposal (0.1) -> 0.1827846364883402, TranlationUpdateProposal (1.0) -> 0.2221112221112221, ShapeUpdateProposal (0.001) -> 0.006664053312426499, RotationUpdateProposal (0.01) -> 0.14262402088772846, ShapeUpdateProposal (0.01) -> 0.3081143517181634)
//      Average Distance: 0.7942234379392421
//      Hausdorff Distance: 5.662372193505642

//      Using large=0.1, medium=0.01, small=0.001
//      3 chains, first landmarks, then twice ASM
//      0.3, shapeUpdateLargeProposal, 0.1, shapeUpdateMediumProposal, 0.3, rotationUpdateProposal, 0.3, translationUpdateProposal
//      0.4, shapeUpdateLargeProposal, 0.2, shapeUpdateMediumProposal, 0.2, rotationUpdateProposal, 0.2, translationUpdateProposal)
//      0.2, shapeUpdateLargeProposal, 0.4, shapeUpdateMediumProposal, 0.2, rotationUpdateProposal, 0.2, translationUpdateProposal)
//      5000 iterations each

//      test 4
//      Map(RotationUpdateProposal (0.01) -> 0.11923963133640553, ShapeUpdateProposal (1.0) -> 6.678539626001781E-4, TranlationUpdateProposal (1.0) -> 0.1926241134751773, ShapeUpdateProposal (0.1) -> 0.0846636259977195)
//      Average Distance: 0.545066910487653
//      Hausdorff Distance: 4.981978388874389

//      test 14
//      Map(RotationUpdateProposal (0.01) -> 0.11051289317086993, ShapeUpdateProposal (1.0) -> 0.0018190086402910413, TranlationUpdateProposal (1.0) -> 0.17775261817152563, ShapeUpdateProposal (0.1) -> 0.09923664122137404)
//      Average Distance: 0.460311320415816
//      Hausdorff Distance: 3.1937226435571135

//      test 23
//      Map(RotationUpdateProposal (0.01) -> 0.1274537695590327, ShapeUpdateProposal (1.0) -> 0.0, TranlationUpdateProposal (1.0) -> 0.18371961560203504, ShapeUpdateProposal (0.1) -> 0.09417169107091587)
//      Average Distance: 0.5741785192090351
//      Hausdorff Distance: 3.951893870903141

//      test 25
//      Map(RotationUpdateProposal (0.01) -> 0.1326644370122631, ShapeUpdateProposal (1.0) -> 0.0, TranlationUpdateProposal (1.0) -> 0.18696397941680962, ShapeUpdateProposal (0.1) -> 0.09036144578313253)
//      Average Distance: 1.100314973070345
//      Hausdorff Distance: 13.050853133532089

//      test 30
//      Map(RotationUpdateProposal (0.01) -> 0.12718064153066966, ShapeUpdateProposal (1.0) -> 4.636068613815484E-4, TranlationUpdateProposal (1.0) -> 0.1784037558685446, ShapeUpdateProposal (0.1) -> 0.10148232611174458)
//      Average Distance: 0.8219450420339894
//      Hausdorff Distance: 6.479588848006874

      val initialParameters = Parameters(EuclideanVector(0, 0, 0), (0.0, 0.0, 0.0),
        DenseVector.zeros[Double](model.rank))

      val logger = new Logger()

      //      val landmarks = modelLmPoints.zip(imgPoints)
      //      val bestTransform: RigidTransformation[_3D] = LandmarkRegistration
      //        .rigid3DLandmarkRegistration(landmarks, center = Point(0, 0, 0))
      //      val alignedModel = model.transform(bestTransform)
      //
      //      val alignedGroup = ui.createGroup("aligned")
      //      val alignedView = ui.show(alignedGroup, alignedModel, "aligned")
      //      alignedView.meshView.color = Color.BLUE
      //
      //      val initialSampleASM = Sample("initial", initialParameters, computeCenterOfMass
      //      (alignedModel.mean))

      val initialSample: Sample = Sample("initial", initialParameters, computeCenterOfMass(model
        .mean))
      val generatorLM = MixtureProposal.fromProposalsWithTransition(
        (0.3, shapeUpdateLargeProposal), (0.1, shapeUpdateMediumProposal),
        (0.3, rotationUpdateProposal), (0.3, translationUpdateProposal)
      )
      val samplesLM = chain("Landmarks", model, initialSample, 5000, generatorLM,
        posteriorEvaluatorLM, logger, modelView, reference)

      val initialSampleASM = samplesLM.maxBy(posteriorEvaluatorLM.logValue)
      val lmView = ui.show(testGroup, model.instance(initialSampleASM.parameters
        .modelCoefficients).transform(initialSampleASM.poseTransformation),
        "after Landmark alignment")
      lmView.color = Color.YELLOW
      val generatorASM = MixtureProposal.fromProposalsWithTransition(
        (0.4, shapeUpdateLargeProposal), (0.2, shapeUpdateMediumProposal),
        (0.2, rotationUpdateProposal), (0.2, translationUpdateProposal)
      )
      val samplesASM = chain("Active Shape Model Large", model, initialSampleASM, 5000,
        generatorASM, posteriorEvaluatorASM, logger, modelView, reference)

      val initialSampleASM2 = samplesASM.maxBy(posteriorEvaluatorASM.logValue)
      val asmView = ui.show(testGroup, model.instance(initialSampleASM2.parameters
        .modelCoefficients).transform(initialSampleASM2.poseTransformation),
        "after first ASM")
      asmView.color = Color.RED
      val generatorASM2 = MixtureProposal.fromProposalsWithTransition(
        (0.2, shapeUpdateLargeProposal), (0.4, shapeUpdateMediumProposal),
        (0.2, rotationUpdateProposal), (0.2, translationUpdateProposal)
      )
      val samplesASM2 = chain("Active Shape Model Small", model, initialSampleASM2, 5000,
        generatorASM2, posteriorEvaluatorASM, logger, modelView, reference)

      println(logger.acceptanceRatios())

      val bestSample = samplesASM2.maxBy(posteriorEvaluatorASM.logValue)
      val bestFit = model.instance(bestSample.parameters.modelCoefficients).transform(bestSample
        .poseTransformation)
      ui.show(testGroup, bestFit, "best fit")

      val dist = scalismo.mesh.MeshMetrics.avgDistance(bestFit, reference)
      val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(bestFit, reference)
      println("Average Distance: " + dist)
      println("Hausdorff Distance: " + hausDist)

      MeshIO.writeMesh(bestFit, new File("data/mcmc/bestFit_" + i + ".stl"))

      StdIn.readLine("Show next fit?")
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

    //    val posteriorFixedPose = model.posterior(correspondences.toIndexedSeq)
    //
    //    for ((id, _, _) <- newCorrespondences) {
    //      val cov = posteriorFixedPose.cov(id, id)
    //      println(s"posterior variance computed by analytic posterior (shape only) for point
    //      with id " +
    //        s"$id = ${cov(0, 0)}, ${cov(1, 1)}, ${cov(2, 2)}")
    //    }
  }

  def computeCenterOfMass(mesh: TriangleMesh[_3D]): Point[_3D] = {
    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector *
      normFactor)
  }

  def getLandmarkLikelihoodEvaluator(model: StatisticalMeshModel, modelLMs: Seq[PointId],
                                     imgPoints: Seq[Point[_3D]]): CachedEvaluator[Sample] = {

    //landmark noise for tests
    val landmarkNoiseVariance = 9.0
    //landmark noise for targets
    //val landmarkNoiseVariance = 15.0
    val uncertainty = MultivariateNormalDistribution(DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance)

    val correspondences = modelLMs.zip(imgPoints).map(modelIdWithTargetPoint => {
      val (modelId, targetPoint) = modelIdWithTargetPoint
      (modelId, targetPoint, uncertainty)
    })

    CachedEvaluator(CorrespondenceEvaluator(model, correspondences))
  }

  def chain(name: String, model: StatisticalMeshModel, initial: Sample, iterations: Int,
            generator: MixtureProposal[Sample] with TransitionProbability[Sample],
            posteriorEvaluator: ProductEvaluator[Sample], logger: Logger,
            view: StatisticalMeshModelViewControls): IndexedSeq[Sample] = {

    chain(name, model, initial, iterations, generator, posteriorEvaluator, logger, view, null)
  }

  def chain(name: String, model: StatisticalMeshModel, initial: Sample, iterations: Int,
            generator: MixtureProposal[Sample] with TransitionProbability[Sample],
            posteriorEvaluator: ProductEvaluator[Sample], logger: Logger,
            view: StatisticalMeshModelViewControls, reference: TriangleMesh[_3D])
  : IndexedSeq[Sample] = {

    val chain: MetropolisHastings[Sample] = MetropolisHastings(generator, posteriorEvaluator)
    val mhIterator = chain.iterator(initial, logger)

    val samplingIterator = for ((sample, iteration) <- mhIterator.zipWithIndex) yield {
      if (iteration % 100 == 0) println(name + ": iteration " + iteration)
      if (iteration % 500 == 0) {
        view.shapeModelTransformationView.shapeTransformationView.coefficients = sample
          .parameters.modelCoefficients
        view.shapeModelTransformationView.poseTransformationView.transformation = sample
          .poseTransformation

        if (reference != null) {
          val dist = scalismo.mesh.MeshMetrics.avgDistance(model.instance(sample.parameters

            .modelCoefficients).transform(sample
            .poseTransformation), reference)
          val hausDist = scalismo.mesh.MeshMetrics.hausdorffDistance(model.instance(sample
            .parameters.modelCoefficients).transform(sample
            .poseTransformation), reference)
          println("Average Distance: " + dist)
          println("Hausdorff Distance: " + hausDist)
        }
      }
      sample
    }

    samplingIterator.take(iterations).toIndexedSeq
  }

  def performChaines() = {

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
      val featureAtPoint: DenseVector[Double] = {
        val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh,
          profile.pointId)
        if (featureAtPoint.isDefined) featureAtPoint.get
        else DenseVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
      }
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

  val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(paramVectorSize),
    DenseMatrix.eye[Double](paramVectorSize) * stddev * stddev)

  implicit val rng = Random(42)

  override def propose(sample: Sample): Sample = {

    val perturbation = perturbationDistr.sample()
    val newParameters = sample.parameters.copy(modelCoefficients = sample.parameters
      .modelCoefficients + perturbationDistr.sample)
    sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample): Double = {

    val residual = to.parameters.modelCoefficients - from.parameters.modelCoefficients
    perturbationDistr.logpdf(residual)
  }
}

case class RotationUpdateProposal(stddev: Double) extends
  ProposalGenerator[Sample] with TransitionProbability[Sample] {

  val perturbationDistr = new MultivariateNormalDistribution(
    DenseVector.zeros[Double](3),
    DenseMatrix.eye[Double](3) * stddev * stddev)

  implicit val rng = Random(42)

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

  override def logTransitionProbability(from: Sample, to: Sample): Double = {
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

  implicit val rng = Random(42)

  def propose(sample: Sample): Sample = {
    val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector
      .fromBreezeVector(perturbationDistr.sample())
    val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
    sample.copy(generatedBy = s"TranlationUpdateProposal ($stddev)", parameters = newParameters)
  }

  override def logTransitionProbability(from: Sample, to: Sample): Double = {
    val residual = to.parameters.translationParameters - from.parameters.translationParameters
    perturbationDistr.logpdf(residual.toBreezeVector)
  }
}

class Logger extends AcceptRejectLogger[Sample] {
  private val numAccepted = collection.mutable.Map[String, Int]()
  private val numRejected = collection.mutable.Map[String, Int]()

  override def accept(current: Sample, sample: Sample, generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]): Unit = {

    val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
    numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)
  }

  override def reject(current: Sample, sample: Sample, generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]): Unit = {

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

