
import './App.css'
import WatchList from './components/watchlist'
import Model_Summary from './components/model';
import 'bootstrap/dist/css/bootstrap.min.css';
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";

function App() {
  return (
    <>
    <BrowserRouter>
    <Routes>
    <Route path="/home" element={<WatchList></WatchList>}/>
    <Route path="/model" element={<Model_Summary></Model_Summary>}></Route>
    </Routes>
    </BrowserRouter>

    </>
  )
}

export default App
