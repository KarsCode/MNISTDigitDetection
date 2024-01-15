import {useState} from 'react';
import './App.css'
import MnistVisualization from './model'
// import MnistVisualization from './model'

function App() {

  const [outputData, setOutputData] = useState({
    randomImage: null as string | null,
    predictedClass: null as number | null,
    isPredictionCorrect: null as boolean | null,
    trainingComplete: null as boolean | null,
  });


  return (
    <div className='flex flex-col justify-center'>
      <div className='flex flex-col gap-2 bg-slate-400 p-5'>
        <div className='text-center text-4xl font-bold'>
          MNIST Handwritten Digits Visualization.
        </div>
        <p className='text-lg'>
          View the console to visualize detailed outputs from the model
        </p>
        </div>

        <div className='flex'>
          <div className='flex flex-col gap-5 w-1/2 border border-black items-center justify-between'>
            <h1 className='text-xl'>Image Input</h1>
            <MnistVisualization onDataUpdate={setOutputData}/>
            <div>
                  {outputData.trainingComplete && (
                  <img src={outputData.randomImage || ''} alt='Random Input' />
                )}
          </div>
          </div>

          <div className='flex flex-col gap-24 w-1/2 border border-black items-center'>
            <h1 className='text-xl'>Image Output</h1>

            <div className='pt-5'>
              {outputData.trainingComplete &&  outputData.isPredictionCorrect && (
                <div className='flex font-bold text-3xl text-green-400'>Predicted Class is: {outputData.predictedClass}</div>
              )}
              {outputData.trainingComplete &&  !outputData.isPredictionCorrect && (
                <div className='flex font-bold text-3xl text-red-500'>Predicted Class is: {outputData.predictedClass}</div>
              )}
              </div>
          </div>

          
        </div>
    </div>
  )
}

export default App
