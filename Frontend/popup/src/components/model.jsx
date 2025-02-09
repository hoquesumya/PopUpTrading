import React from "react"
import "../styles/model.css"
import { useLocation } from 'react-router-dom'
import { useEffect, useState } from "react";
import Nav from 'react-bootstrap/Nav';
import { Link } from "react-router-dom";
import axios from "axios";
import Navbar from 'react-bootstrap/Navbar';
import { Button } from "react-bootstrap";

function Model_Summary(){
    const location = useLocation()
    const tick  = location.state
    const [hasFetched, setHasFetched] = useState(false);
    const ticker = tick["ticker_type"]
    useEffect(() => {
        const fetch_data = async () =>{
            try {
                const response = await axios.get("http://127.0.0.1:8000/Sentiment");
                console.log("res",response)
                setHasFetched(true)

            }catch (err) {   
                console.error("Error occurred:", err.response); // Log full error response
                console.error("Error message:", err.message);
                console.error("Error stack:", err.stack);
            }
        }
       if (ticker == "SPY" && !hasFetched){
            fetch_data()
       }
    }, [ticker,hasFetched]); 
    console.log(ticker)

    return(
        <>
        <div  className = {`${ticker}_model main_details`} >
        <Navbar expand="xl" fixed="top" style={{paddingLeft:"15px", paddingTop:"20px"}}>
                <Nav>
                    <Link  to={'/home'}>
                    <Button style={{backgroundColor:"#4E6E4D", border:"none"}}>
                        Home
                    </Button>
                    </Link>
                </Nav>
        </Navbar>
        <div className="main_model_details"> 
            {ticker == "SPY" && <>SPY</>}
        </div>

        </div>
        </>

    )
}
export default Model_Summary