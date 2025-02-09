import React from "react"
import { Row } from "react-bootstrap"
import { Button } from "react-bootstrap"
import "../styles/watchlistMain.css"
import NotificationsNoneIcon from '@mui/icons-material/NotificationsNone';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
function WathcListMain(){
    var ticker_list = ["SPY", "VOO", "IVV", 'VTI', "QQQ"]
    return(
        <div className="ticker_list">
            <Row className="md-6 ticker_list_child align-items-stretch">
            {ticker_list.map((ticker, index) => (
                <Row key={index} className={`flex-fill border ticker-child-${index}`}
                    style ={{
                        display:"grid"
                    }}
                >
                    <div className ={`ticker-child-child${index}`}
                    style={{
                        margin:"auto",
                        display:"flex"
                    }
                    }
                    >
                        <div className="p-2 flex-grow-1 bd-highlight" 
                            style={{
                               fontWeight:900,
                               margin:"auto"
                            }}
                        > {ticker}</div>
                        <Button className="p-3" style={{backgroundColor:"transparent",  marginRight: "10px"}}>
                            $0.7
                        </Button> 
                        <Button className="p-3" style={{backgroundColor:"transparent", border:"none"}}>
                        <NotificationsNoneIcon />
                        </Button>
                       
                
                    </div>
                    <div
                    style={{
                        display:"flex",
                        flexDirection:"row",
                        justifyContent:"center",
                        alignItems:"center"
                    }}
                    >
                        <Button style={{backgroundColor:"transparent", border:"none"}}>
                        <KeyboardArrowDownIcon></KeyboardArrowDownIcon>

                        </Button>
                           
                    </div>
                </Row>
            ))}
                
            </Row>

        </div>
    )
}
export default WathcListMain